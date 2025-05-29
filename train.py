import torch
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ComposerUnet import ComposerDataset
from ComposerPipeline import ComposerStableDiffusionPipeline

if __name__ == '__main__':
    composer_pipe = ComposerStableDiffusionPipeline.load_custom_pretrained(load_directory=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8  # 设置批次大小

    composer_pipe.to(device)
    total_batch_size = batch_size

    # 创建数据加载器
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

    # 加载数据集
    train_dataloader = DataLoader(dataset, batch_size=total_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=total_batch_size, shuffle=False)
    num_epochs = 100
    max_steps = len(train_dataloader) * num_epochs

    # 简化训练循环
    optimizer = torch.optim.AdamW(composer_pipe.unet.parameters(), lr=1e-5)

    # 学习率线性增长
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=max_steps
    )
    composer_pipe.set_progress_bar_config(disable=True)  # 禁用进度条显示

    # 设置为训练模式
    composer_pipe.unet.train()
    composer_pipe.vae.eval()
    scaling_factor = composer_pipe.vae.config.scaling_factor
    num_train_steps = 1000

    # 训练循环
    print("Starting training...")

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # 训练循环
        composer_pipe.unet.train()
        for batch in train_dataloader:
            # 前向调用需要传递所有条件
            images = batch["image"].to(device=device)

            # 冻结VAE
            with torch.no_grad():
                latents = composer_pipe.vae.encode(images).latent_dist.sample() * scaling_factor

            # 添加噪声
            noise = torch.randn_like(latents)

            # 计算噪声
            timesteps = torch.randint(0, num_train_steps, (latents.shape[0],))
            timesteps = timesteps.to(device=device)
            timesteps.long()

            # 添加噪声到潜在变量
            noisy_latents = composer_pipe.scheduler.add_noise(latents, noise, timesteps)

            # 计算噪声预测
            noise_pred = composer_pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=None,  # 使用CLIP生成的新context
                image=batch["pixel_values"],
                prompt=batch["prompt"],
                color=batch["color"],
                sketch=batch["sketch"],
                instance=batch["instance"],
                depth=batch["depth"],
                intensity=batch["intensity"]
            )[0]

            # 计算损失
            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")

        # 反向传播
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(composer_pipe.unet.parameters(), 1.0)
        # 更新参数
        optimizer.step()
        # 学习率调度
        lr_scheduler.step()
        # 清除梯度
        optimizer.zero_grad()
        print(f"Train Loss: {loss.item():.4f}")

        # 验证循环
        if (epoch + 1) % 10 == 0:  # 每10个epoch验证一次
            composer_pipe.unet.eval()
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_images = val_batch["image"].to(device)
                    val_latents = composer_pipe.vae.encode(val_images).latent_dist.sample() * scaling_factor
                    val_noise = torch.randn_like(val_latents)
                    val_timesteps = torch.randint(0, num_train_steps, (val_latents.shape[0],)).to(device).long()
                    val_noisy_latents = composer_pipe.scheduler.add_noise(val_latents, val_noise, val_timesteps)

                    # 验证噪声预测
                    val_noise_pred = composer_pipe.unet(
                        val_noisy_latents,
                        val_timesteps,
                        encoder_hidden_states=None,
                        image=val_batch["pixel_values"],
                        prompt=val_batch["prompt"],
                        color=val_batch["color"],
                        sketch=val_batch["sketch"],
                        instance=val_batch["instance"],
                        depth=val_batch["depth"],
                        intensity=val_batch["intensity"]
                    )[0]

                    # Apply classifier-free guidance
                    val_noise_pred_uncond, val_noise_pred_cond = val_noise_pred.chunk(2, dim=0)
                    val_noise_pred = val_noise_pred_uncond + 7.5 * (val_noise_pred_cond - val_noise_pred_uncond)

                    # 计算验证损失
                    val_loss = torch.nn.functional.mse_loss(val_noise_pred, val_noise, reduction="mean")
                    print(f"Validation Loss: {val_loss.item():.4f}")

    print("Training complete!")
    # 保存模型
    save_directory = "./ComposerStableDiffusion"
    composer_pipe.save_custom_pretrained(save_directory)
    print(f"Model saved to{save_directory}")
