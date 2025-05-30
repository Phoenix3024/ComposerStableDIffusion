import torch
from torch import nn
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import os

from ComposerUnet import ComposerDataset
from ComposerPipeline import ComposerStableDiffusionPipeline


def main():
    # 检测可用GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s). Using DataParallel for training.")

    # 加载模型
    composer_pipe = ComposerStableDiffusionPipeline.load_custom_pretrained(load_directory=None)

    # 设置主设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    composer_pipe.to(device)
    print(f"Pipeline loaded and moved to {device}.")

    # 使用DataParallel包装UNet
    if num_gpus > 1:
        composer_pipe = nn.DataParallel(composer_pipe)
    composer_pipe = composer_pipe.module if isinstance(composer_pipe, nn.DataParallel) else composer_pipe

    # 训练参数
    batch_size = 32 * num_gpus  # 总batch size = 单卡batch * GPU数量
    num_epochs = 100
    num_train_steps = 1000
    validation_interval = 3  # 每3个epoch验证一次
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

    # 数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 * num_gpus,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 * num_gpus,
        pin_memory=True
    )

    # 优化器和学习率调度
    optimizer = torch.optim.AdamW(composer_pipe.unet.parameters(), lr=1e-5 * num_gpus)  # 线性缩放学习率
    max_steps = len(train_dataloader) * num_epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=max_steps
    )

    # 训练循环
    main_pbar = tqdm(total=num_epochs, desc="Total Training Progress")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        composer_pipe.unet.train()
        epoch_pbar = tqdm(total=len(train_dataloader),
                          desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                          leave=False)

        total_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # 数据准备
            images = batch["image"].to(device, non_blocking=True)
            prompt = batch["prompt"]
            color = batch["color"].to(device, non_blocking=True)
            sketch = batch["sketch"].to(device, non_blocking=True)
            instance = batch["instance"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            intensity = batch["intensity"].to(device, non_blocking=True)

            # 第一部分：CLIP融合条件

            # 处理文本
            text_inputs = composer_pipe.tokenizer(text=prompt, return_tensors="pt", padding="max_length",
                                                  max_length=77, truncation=True).to(device)
            text_features = composer_pipe.text_encoder(text_inputs.input_ids.to(device))[0]

            # 处理颜色
            color = color.to(device)

            # 提取特征
            with torch.no_grad():

                latents = composer_pipe.vae.encode(images).latent_dist.sample()
                cond_sketch = composer_pipe.vae.encode(sketch).latent_dist.sample()
                cond_instance = composer_pipe.vae.encode(instance).latent_dist.sample()
                cond_depth = composer_pipe.vae.encode(depth).latent_dist.sample()
                cond_intensity = composer_pipe.vae.encode(intensity).latent_dist.sample()

                latents = latents * composer_pipe.vae.config.scaling_factor
                cond_sketch = cond_sketch * composer_pipe.vae.config.scaling_factor
                cond_instance = cond_instance * composer_pipe.vae.config.scaling_factor
                cond_depth = cond_depth * composer_pipe.vae.config.scaling_factor
                cond_intensity = cond_intensity * composer_pipe.vae.config.scaling_factor

            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, num_train_steps, (latents.shape[0],), device=device).long()
            noisy_latents = composer_pipe.scheduler.add_noise(latents, noise, timesteps)

            # 处理图像和文本特征
            B = text_features.shape[0]
            text_features = composer_pipe.clip_text_proj(text_features)
            color_features = composer_pipe.color_proj(color)
            color_features = color_features.view(B, 4, 768)
            clip_context = torch.cat([text_features, color_features], dim=1)

            # 第二部分：时间步融合
            clip_text_time_emb = composer_pipe.clip_text_time_proj(text_features).view(B, 320)
            color_time_emb = composer_pipe.color_time_proj(color_features).view(B, 320)
            timestep_cond = clip_text_time_emb + color_time_emb

            # 第三部分：视觉条件处理

            condition_features = composer_pipe.local_condition_proj(
                sketch=cond_sketch,
                instance=cond_instance,
                depth=cond_depth,
                intensity=cond_intensity
            )
            local_condition = torch.zeros_like(noisy_latents)
            for i in range(len(condition_features)):
                local_condition += condition_features[i]

            # 第四部分：输入层融合

            combined_input = torch.cat([noisy_latents, local_condition], dim=1)

            # 前向传播
            noise_pred = composer_pipe.unet(
                combined_input,
                timesteps,
                encoder_hidden_states=clip_context,
                timestep_cond=timestep_cond
            )[0]

            # 计算损失
            loss = nn.functional.mse_loss(noise_pred, noise)
            total_loss += loss.item()

            # 反向传播
            loss.backward()
            nn.utils.clip_grad_norm_(composer_pipe.unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            epoch_pbar.set_postfix({
                'batch_loss': f"{loss.item():.4f}",
                'avg_loss': f"{avg_loss:.4f}",
                'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
            })
            epoch_pbar.update()

        epoch_pbar.close()
        main_pbar.update(1)
        print(f"Epoch {epoch + 1} Train Loss: {avg_loss:.4f}")

        # 验证阶段
        if (epoch + 1) % validation_interval == 0:
            composer_pipe.unet.eval()
            val_loss = 0.0
            val_samples = 0

            val_pbar = tqdm(total=len(val_dataloader), desc=f"Validation Epoch {epoch + 1}", leave=False)

            with torch.no_grad():
                for val_batch in val_dataloader:
                    # 数据准备
                    val_images = val_batch["image"].to(device, non_blocking=True)
                    val_color = val_batch["color"].to(device, non_blocking=True)
                    val_prompt = val_batch["prompt"]
                    val_sketch = val_batch["sketch"].to(device, non_blocking=True)
                    val_instance = val_batch["instance"].to(device, non_blocking=True)
                    val_depth = val_batch["depth"].to(device, non_blocking=True)
                    val_intensity = val_batch["intensity"].to(device, non_blocking=True)

                    # VAE编码
                    val_latents = composer_pipe.vae.encode(val_images).latent_dist.sample()
                    val_cond_text = composer_pipe.tokenizer(text=val_prompt, return_tensors="pt", padding="max_length",
                                                            max_length=77, truncation=True).to(device)
                    val_cond_text_features = composer_pipe.text_encoder(val_cond_text.input_ids.to(device))[0]
                    val_cond_color = composer_pipe.color_proj(val_color).view(val_color.shape[0], 4, 768)
                    val_cond_sketch = composer_pipe.vae.encode(val_sketch).latent_dist.sample()
                    val_cond_instance = composer_pipe.vae.encode(val_instance).latent_dist.sample()
                    val_cond_depth = composer_pipe.vae.encode(val_depth).latent_dist.sample()
                    val_cond_intensity = composer_pipe.vae.encode(val_intensity).latent_dist.sample()

                    val_latents = val_latents * composer_pipe.vae.config.scaling_factor
                    val_cond_sketch = val_cond_sketch * composer_pipe.vae.config.scaling_factor
                    val_cond_instance = val_cond_instance * composer_pipe.vae.config.scaling_factor
                    val_cond_depth = val_cond_depth * composer_pipe.vae.config.scaling_factor
                    val_cond_intensity = val_cond_intensity * composer_pipe.vae.config.scaling_factor

                    # 添加噪声
                    val_noise = torch.randn_like(val_latents)
                    val_timesteps = torch.randint(0, num_train_steps, (val_latents.shape[0],), device=device).long()
                    val_noisy_latents = composer_pipe.scheduler.add_noise(val_latents, val_noise, val_timesteps)
                    val_time_cond_text = composer_pipe.clip_text_time_proj(val_cond_text_features).view(
                        val_cond_text_features.shape[0], 320
                    )
                    val_time_cond_color = composer_pipe.color_time_proj(val_cond_color).view(
                        val_cond_color.shape[0], 320
                    )
                    val_timesteps_cond = val_time_cond_text + val_time_cond_color

                    # 准备条件输入
                    val_cond_inputs = {
                        "prompt": val_cond_text_features,
                        "color": val_cond_color,
                        "time_cond": val_timesteps_cond,
                        "sketch": val_cond_sketch,
                        "instance": val_cond_instance,
                        "depth": val_cond_depth,
                        "intensity": val_cond_intensity
                    }

                    # 前向传播（带guidance）
                    # 复制输入用于无条件和有条件预测
                    duplicated_noisy_latents = torch.cat([val_noisy_latents] * 2)
                    duplicated_timesteps = torch.cat([val_timesteps] * 2)

                    # 复制条件输入
                    duplicated_cond_inputs = {}
                    for k, v in val_cond_inputs.items():
                        if isinstance(v, torch.Tensor):
                            duplicated_cond_inputs[k] = torch.cat([v] * 2)
                        else:
                            duplicated_cond_inputs[k] = v * 2  # 对于非tensor，简单复制

                    val_local_condition = composer_pipe.local_condition_proj(
                        sketch=duplicated_cond_inputs["sketch"],
                        instance=duplicated_cond_inputs["instance"],
                        depth=duplicated_cond_inputs["depth"],
                        intensity=duplicated_cond_inputs["intensity"]
                    )
                    duplicated_noisy_latents = torch.cat([duplicated_noisy_latents, val_local_condition], dim=1)
                    val_encoder_hidden_states = torch.cat([
                        duplicated_cond_inputs["prompt"],
                        duplicated_cond_inputs["color"]
                    ], dim=1)

                    # 模型前向
                    noise_pred = composer_pipe.unet(
                        duplicated_noisy_latents,
                        duplicated_timesteps,
                        encoder_hidden_states=val_encoder_hidden_states,
                        timestep_cond=val_timesteps_cond
                    )[0]

                    # 拆分预测结果
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # 计算损失
                    batch_loss = nn.functional.mse_loss(noise_pred, val_noise)

                    val_loss += batch_loss.item() * val_images.size(0)
                    val_samples += val_images.size(0)

                    val_pbar.set_postfix({'val_batch_loss': f"{batch_loss.item():.4f}"})
                    val_pbar.update()

            val_pbar.close()
            avg_val_loss = val_loss / val_samples
            print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"🌟 New best validation loss: {avg_val_loss:.4f}, saving model...")

                # 保存UNet参数
                save_directory = "./ComposerStableDiffusion"
                os.makedirs(save_directory, exist_ok=True)

                if num_gpus > 1:
                    torch.save(composer_pipe.unet.module.state_dict(),
                               os.path.join(save_directory, "unet_best.pth"))
                else:
                    torch.save(composer_pipe.unet.state_dict(),
                               os.path.join(save_directory, "unet_best.pth"))

                # 保存完整pipeline（包含VAE等组件）
                composer_pipe.save_custom_pretrained(save_directory)
                print(f"✅ Best model saved to {save_directory}")
            else:
                print(f"🚫 Current validation loss {avg_val_loss:.4f} not better than best {best_val_loss:.4f}")
        main_pbar.update(1)


if __name__ == '__main__':
    main()
    print("Training completed!")
