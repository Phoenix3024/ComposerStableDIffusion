import datetime
import logging
import os

import torch
from diffusers.optimization import get_scheduler
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ComposerPipeline import ComposerStableDiffusionPipeline
from ComposerUnet import ComposerDataset


# 设置日志
def setup_logging(exp_name):
    # 创建日志目录
    log_dir = f"logs/{exp_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)

    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler()
        ]
    )

    # 创建TensorBoard写入器
    tb_writer = SummaryWriter(log_dir=log_dir)

    return log_dir, tb_writer


def validation_step(model, val_dataloader, device, guidance_scale=7.5):
    """执行验证步骤，包括CFG推理"""
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model.module.unet.eval()
        model.module.clip_image_proj.eval()
        model.module.color_proj.eval()
        model.module.clip_image_time_proj.eval()
        model.module.clip_text_time_proj.eval()
        model.module.color_time_proj.eval()
        model.module.local_condition_proj.eval()
        model.module.vae.eval()
        model.module.text_encoder.eval()
        model.module.image_encoder.eval()
    else:
        model.unet.eval()
        model.clip_image_proj.eval()
        model.color_proj.eval()
        model.clip_image_time_proj.eval()
        model.clip_text_time_proj.eval()
        model.color_time_proj.eval()
        model.local_condition_proj.eval()
        model.vae.eval()
        model.text_encoder.eval()
        model.image_encoder.eval()

    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        val_pbar = tqdm(val_dataloader, desc="Validation", leave=False)
        for batch in val_pbar:
            # 准备数据
            image = batch["image"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            prompt = batch["prompt"]
            color = batch["color"].to(device, non_blocking=True)
            sketch = batch["sketch"].to(device, non_blocking=True)
            instance = batch["instance"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            intensity = batch["intensity"].to(device, non_blocking=True)

            # 获取目标图像的潜在表示
            image_latents = model.vae.encode(image).latent_dist.sample()
            image_latents = image_latents * model.vae.config.scaling_factor

            # 添加噪声
            noise = torch.randn_like(image_latents)
            timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps,
                                      (image_latents.shape[0],), device=device).long()

            noisy_latents = model.scheduler.add_noise(image_latents, noise, timesteps)

            # 处理图像条件
            clip_image_embeds = model.image_encoder(pixel_values).image_embeds

            # 处理文本条件
            text_inputs = model.tokenizer(
                prompt,
                padding="max_length",
                max_length=model.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            text_embeddings = model.text_encoder(text_inputs.input_ids)[0]

            # 处理颜色条件
            B = color.shape[0]
            clip_image_embeds = model.clip_image_proj(clip_image_embeds).view(B, 4, 768)
            color_emb = model.color_proj(color).view(B, 4, 768)

            # 准备时间条件
            time_cond_pixel_values = model.clip_image_time_proj(clip_image_embeds).view(B, 320)
            time_cond_prompt = model.clip_text_time_proj(text_embeddings).view(B, 320)
            time_cond_color = model.color_time_proj(color_emb).view(B, 320)
            time_cond = time_cond_pixel_values + time_cond_prompt + time_cond_color

            # 编码视觉条件
            def encode_visual_condition(cond):
                encoded = model.vae.encode(cond).latent_dist.sample()
                return encoded * model.vae.config.scaling_factor

            sketch_enc = encode_visual_condition(sketch)
            instance_enc = encode_visual_condition(instance)
            depth_enc = encode_visual_condition(depth)
            intensity_enc = encode_visual_condition(intensity)

            # 处理视觉条件
            cond_local_conditions = model.local_condition_proj(
                sketch=sketch_enc,
                instance=instance_enc,
                depth=depth_enc,
                intensity=intensity_enc
            )

            # 合并视觉条件
            local_conditions = torch.zeros_like(noisy_latents)
            for cond in cond_local_conditions:
                local_conditions += cond

            # 将视觉条件与噪声潜在空间合并
            model_input = torch.cat([noisy_latents, local_conditions], dim=1)

            # 准备条件嵌入
            encoder_hidden_states = torch.cat([clip_image_embeds, text_embeddings, color_emb], dim=1)

            # 预测噪声
            noise_pred = model.unet(
                model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=time_cond
            ).sample

            # 计算损失
            loss = nn.functional.mse_loss(noise_pred, noise)
            val_loss += loss.item()
            num_batches += 1

            val_pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

    avg_val_loss = val_loss / num_batches if num_batches > 0 else float('inf')
    return avg_val_loss


def main():
    # 实验名称
    exp_name = "composer_sd_v1"

    # 设置日志和TensorBoard
    log_dir, tb_writer = setup_logging(exp_name)
    logging.info(f"Starting experiment: {exp_name}")
    logging.info(f"Log directory: {log_dir}")

    # 检测可用GPU数量
    num_gpus = torch.cuda.device_count()
    logging.info(f"Found {num_gpus} GPU(s). Using DataParallel for training.")

    # 加载模型
    model = ComposerStableDiffusionPipeline.load_custom_pretrained(load_directory=None)

    # 设置主设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 将模型移到设备
    for module in [model.unet, model.clip_image_proj, model.color_proj, model.clip_image_time_proj,
                   model.clip_text_time_proj, model.color_time_proj, model.local_condition_proj]:
        module.to(device)
    logging.info(f"Pipeline loaded and moved to {device}.")

    # 冻结不需要训练的模块
    for param in model.vae.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # 使用DataParallel包装模型
    if num_gpus > 1:
        model = nn.DataParallel(model)
        logging.info(f"Using DataParallel on {num_gpus} GPUs.")

    # 训练参数
    batch_size = 32 * num_gpus  # 每个GPU的batch size
    num_epochs = 100
    validation_interval = 1  # 每1个epoch验证一次
    guidance_scale = 7.5  # Classifier-free guidance系数

    # 数据集
    dataset = ComposerDataset(
        num_samples=123403,
        unlabeled_dir="data/unlabeled2017",
        feature_dir="data/feature_maps",
        caption_csv="data/image_captions.csv",
        filenames_npy="data/color_features/filenames.npy",
        color_npy="data/color_features/color_histograms.npy"
    )

    # 数据集分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    logging.info(f"Dataset sizes - Train: {train_size}, Validation: {val_size}")

    # 数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 * num_gpus,
        pin_memory=True,
        drop_last=True  # 确保批次完整
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 * num_gpus,
        pin_memory=True
    )

    # 优化器和学习率调度
    # 设置优化器 - 只训练新增模块
    optimizer = torch.optim.AdamW([
        {'params': model.module.unet.parameters() if num_gpus > 1 else model.unet.parameters()},
        {'params': model.module.clip_image_proj.parameters() if num_gpus > 1 else model.clip_image_proj.parameters()},
        {'params': model.module.color_proj.parameters() if num_gpus > 1 else model.color_proj.parameters()},
        {
            'params': model.module.clip_image_time_proj.parameters() if num_gpus > 1 else model.clip_image_time_proj.parameters()},
        {
            'params': model.module.clip_text_time_proj.parameters() if num_gpus > 1 else model.clip_text_time_proj.parameters()},
        {'params': model.module.color_time_proj.parameters() if num_gpus > 1 else model.color_time_proj.parameters()},
        {
            'params': model.module.local_condition_proj.parameters() if num_gpus > 1 else model.local_condition_proj.parameters()}
    ], lr=1e-5 * num_gpus, weight_decay=1e-4)

    max_steps = len(train_dataloader) * num_epochs
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=max_steps
    )

    # 训练循环
    best_val_loss = float('inf')
    global_step = 0

    logging.info("Starting training...")
    for epoch in range(num_epochs):
        # 训练阶段
        if num_gpus > 1:
            model.module.unet.train()
            model.module.clip_image_proj.train()
            model.module.color_proj.train()
            model.module.clip_image_time_proj.train()
            model.module.clip_text_time_proj.train()
            model.module.color_time_proj.train()
            model.module.local_condition_proj.train()
        else:
            model.unet.train()
            model.clip_image_proj.train()
            model.color_proj.train()
            model.clip_image_time_proj.train()
            model.clip_text_time_proj.train()
            model.color_time_proj.train()
            model.local_condition_proj.train()

        epoch_train_loss = 0.0
        epoch_pbar = tqdm(total=len(train_dataloader),
                          desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                          leave=False)

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # 数据准备
            image = batch["image"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            prompt = batch["prompt"]
            color = batch["color"].to(device, non_blocking=True)
            sketch = batch["sketch"].to(device, non_blocking=True)
            instance = batch["instance"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            intensity = batch["intensity"].to(device, non_blocking=True)

            # 获取目标图像的潜在表示
            with torch.no_grad():
                image_latents = model.module.vae.encode(
                    image).latent_dist.sample() if num_gpus > 1 else model.vae.encode(image).latent_dist.sample()
                image_latents = image_latents * (
                    model.module.vae.config.scaling_factor if num_gpus > 1 else model.vae.config.scaling_factor)

            # 添加噪声
            noise = torch.randn_like(image_latents)
            timesteps = torch.randint(0,
                                      model.module.scheduler.config.num_train_timesteps if num_gpus > 1 else model.scheduler.config.num_train_timesteps,
                                      (image_latents.shape[0],), device=device).long()

            noisy_latents = (model.module.scheduler if num_gpus > 1 else model.scheduler).add_noise(image_latents,
                                                                                                    noise, timesteps)

            # 第一部分：CLIP融合条件
            with torch.no_grad():
                # 处理图像条件
                clip_image_embeds = (
                    model.module.image_encoder(pixel_values).image_embeds if num_gpus > 1 else model.image_encoder(
                        pixel_values).image_embeds)

                # 处理文本条件
                text_inputs = (model.module.tokenizer if num_gpus > 1 else model.tokenizer)(
                    prompt,
                    padding="max_length",
                    max_length=(
                        model.module.tokenizer.model_max_length if num_gpus > 1 else model.tokenizer.model_max_length),
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                text_embeddings = (model.module.text_encoder(text_inputs.input_ids)[0] if num_gpus > 1 else
                                   model.text_encoder(text_inputs.input_ids)[0])
            # 处理颜色条件
            B = color.shape[0]
            clip_image_embeds = (model.module.clip_image_proj(clip_image_embeds).view(B, 4,
                                                                                      768) if num_gpus > 1 else model.clip_image_proj(
                clip_image_embeds).view(B, 4, 768))
            color_emb = (model.module.color_proj(color).view(B, 4, 768) if num_gpus > 1 else model.color_proj(
                color).view(B, 4, 768))

            # 准备时间条件
            time_cond_pixel_values = (model.module.clip_image_time_proj(clip_image_embeds).view(B,
                                                                                                320) if num_gpus > 1 else model.clip_image_time_proj(
                clip_image_embeds).view(B, 320))
            time_cond_prompt = (model.module.clip_text_time_proj(text_embeddings).view(B,
                                                                                       320) if num_gpus > 1 else model.clip_text_time_proj(
                text_embeddings).view(B, 320))
            time_cond_color = (
                model.module.color_time_proj(color_emb).view(B, 320) if num_gpus > 1 else model.color_time_proj(
                    color_emb).view(B, 320))
            time_cond = time_cond_pixel_values + time_cond_prompt + time_cond_color

            # 编码视觉条件
            with torch.no_grad():
                def encode_visual_condition(cond):
                    encoded = (model.module.vae.encode(cond).latent_dist.sample() if num_gpus > 1 else model.vae.encode(
                        cond).latent_dist.sample())
                    return encoded * (
                        model.module.vae.config.scaling_factor if num_gpus > 1 else model.vae.config.scaling_factor)

                sketch_enc = encode_visual_condition(sketch)
                instance_enc = encode_visual_condition(instance)
                depth_enc = encode_visual_condition(depth)
                intensity_enc = encode_visual_condition(intensity)

            # 处理视觉条件
            cond_local_conditions = (model.module.local_condition_proj(
                sketch=sketch_enc,
                instance=instance_enc,
                depth=depth_enc,
                intensity=intensity_enc
            ) if num_gpus > 1 else model.local_condition_proj(
                sketch=sketch_enc,
                instance=instance_enc,
                depth=depth_enc,
                intensity=intensity_enc
            ))

            # 合并视觉条件
            local_conditions = torch.zeros_like(noisy_latents)
            for cond in cond_local_conditions:
                local_conditions += cond

            # 将视觉条件与噪声潜在空间合并
            model_input = torch.cat([noisy_latents, local_conditions], dim=1)

            # 准备条件嵌入
            encoder_hidden_states = torch.cat([clip_image_embeds, text_embeddings, color_emb], dim=1)

            # 预测噪声
            noise_pred = (model.module.unet(
                model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=time_cond
            ).sample if num_gpus > 1 else model.unet(
                model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=time_cond
            ).sample)

            # 计算损失
            loss = nn.functional.mse_loss(noise_pred, noise)
            epoch_train_loss += loss.item()

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.module.unet.parameters() if num_gpus > 1 else model.unet.parameters(),
                                           1.0)
            torch.nn.utils.clip_grad_norm_(
                model.module.clip_image_proj.parameters() if num_gpus > 1 else model.clip_image_proj.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(
                model.module.color_proj.parameters() if num_gpus > 1 else model.color_proj.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(
                model.module.clip_image_time_proj.parameters() if num_gpus > 1 else model.clip_image_time_proj.parameters(),
                1.0)
            torch.nn.utils.clip_grad_norm_(
                model.module.clip_text_time_proj.parameters() if num_gpus > 1 else model.clip_text_time_proj.parameters(),
                1.0)
            torch.nn.utils.clip_grad_norm_(
                model.module.color_time_proj.parameters() if num_gpus > 1 else model.color_time_proj.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(
                model.module.local_condition_proj.parameters() if num_gpus > 1 else model.local_condition_proj.parameters(),
                1.0)

            # 更新优化器
            optimizer.step()
            lr_scheduler.step()
            global_step += 1

            # 更新进度条
            avg_loss = epoch_train_loss / (batch_idx + 1)
            epoch_pbar.set_postfix({
                'batch_loss': f"{loss.item():.4f}",
                'avg_loss': f"{avg_loss:.4f}",
                'lr': f"{lr_scheduler.get_last_lr()[0]}"
            })
            epoch_pbar.update()

            # 记录TensorBoard
            tb_writer.add_scalar('train/batch_loss', loss.item(), global_step)
            tb_writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], global_step)

        epoch_pbar.close()
        avg_epoch_loss = epoch_train_loss / len(train_dataloader)
        tb_writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
        logging.info(f"Epoch {epoch + 1} Train Loss: {avg_epoch_loss:.4f}")

        # 验证阶段
        if (epoch + 1) % validation_interval == 0 or epoch == num_epochs - 1:
            logging.info(f"Running validation after epoch {epoch + 1}")
            val_loss = validation_step(
                model.module if num_gpus > 1 else model,
                val_dataloader,
                device,
                guidance_scale
            )
            tb_writer.add_scalar('val/loss', val_loss, epoch)
            logging.info(f"Validation Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(log_dir, "best_model")
                if num_gpus > 1:
                    model.module.save_custom_pretrained(save_path)
                else:
                    model.save_custom_pretrained(save_path)
                logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch + 1}")
            if num_gpus > 1:
                model.module.save_custom_pretrained(checkpoint_path)
            else:
                model.save_custom_pretrained(checkpoint_path)
            logging.info(f"Checkpoint saved at epoch {epoch + 1}")

    # 最终模型保存
    final_save_path = os.path.join(log_dir, "final_model")
    if num_gpus > 1:
        model.module.save_custom_pretrained(final_save_path)
    else:
        model.save_custom_pretrained(final_save_path)
    logging.info(f"Final model saved to {final_save_path}")

    # 关闭TensorBoard写入器
    tb_writer.close()


if __name__ == '__main__':
    main()
    logging.info("Training completed!")