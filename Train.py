import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from PipelineTest import ComposerStableDiffusionPipeline


# 定义模拟数据集
class RandomConditionDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.image_size = 512
        self.condition_size = 512
        self.color_dim = 156

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": torch.rand(3, self.image_size, self.image_size),
            "pixel_values": torch.rand(3, 224, 224),
            "prompt": "A fantasy landscape",  # 所有样本使用相同提示
            "color": torch.rand(self.color_dim),
            "sketch": torch.rand(3, self.condition_size, self.condition_size),
            "instance": torch.rand(3, self.condition_size, self.condition_size),
            "depth": torch.rand(3, self.condition_size, self.condition_size),
            "intensity": torch.rand(3, self.condition_size, self.condition_size)
        }


# 训练函数
def train_composer_model(model, num_epochs=10, batch_size=2, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据集和数据加载器
    dataset = RandomConditionDataset(num_samples=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 将模型移到设备
    for module in [model.unet, model.clip_image_proj, model.color_proj, model.clip_image_time_proj,
                   model.clip_text_time_proj, model.color_time_proj, model.local_condition_proj]:
        module.to(device)

    # 设置优化器 - 只训练新增模块
    optimizer = optim.AdamW([
        {'params': model.unet.parameters()},
        {'params': model.clip_image_proj.parameters()},
        {'params': model.color_proj.parameters()},
        {'params': model.clip_image_time_proj.parameters()},
        {'params': model.clip_text_time_proj.parameters()},
        {'params': model.color_time_proj.parameters()},
        {'params': model.local_condition_proj.parameters()}
    ], lr=learning_rate)

    # 损失函数
    criterion = nn.MSELoss()

    # 训练循环
    model.unet.train()
    model.clip_image_proj.train()
    model.color_proj.train()
    model.clip_image_time_proj.train()
    model.clip_text_time_proj.train()
    model.color_time_proj.train()
    model.local_condition_proj.train()
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            # 准备输入数据
            image = batch["image"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            prompt = batch["prompt"]
            color = batch["color"].to(device)
            sketch = batch["sketch"].to(device)
            instance = batch["instance"].to(device)
            depth = batch["depth"].to(device)
            intensity = batch["intensity"].to(device)

            # 获取目标图像的潜在表示
            with torch.no_grad():
                image_latents = model.vae.encode(image).latent_dist.sample()
                image_latents = image_latents * model.vae.config.scaling_factor

            # 添加噪声
            noise = torch.randn_like(image_latents)
            timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps,
                                      (image_latents.shape[0],), device=device).long()

            noisy_latents = model.scheduler.add_noise(image_latents, noise, timesteps)

            # 前向传播
            # 准备条件输入
            with torch.no_grad():
                # 处理图像条件
                clip_image_embeds = model.image_encoder(pixel_values).image_embeds
                # print(f"Image embeddings shape: {clip_image_embeds.shape}")

                # 处理文本条件
                text_inputs = model.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=model.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                text_embeddings = model.text_encoder(text_inputs.input_ids)[0]
                # print(f"Text embeddings shape: {text_embeddings.shape}")

            # 处理颜色条件
            B = color.shape[0]
            clip_image_embeds = model.clip_image_proj(clip_image_embeds).view(B, 4, 768)
            color_emb = model.color_proj(color).view(B, 4, 768)
            # print(f"Clip image embeddings shape: {clip_image_embeds.shape}")
            # print(f"Color embeddings shape: {color_emb.shape}")

            # 准备时间条件
            time_cond_pixel_values = model.clip_image_time_proj(clip_image_embeds).view(B, 320)
            time_cond_prompt = model.clip_text_time_proj(text_embeddings).view(B, 320)
            time_cond_color = model.color_time_proj(color_emb).view(B, 320)
            time_cond = time_cond_pixel_values + time_cond_prompt + time_cond_color

            # 编码视觉条件
            with torch.no_grad():
                def encode_visual_condition(cond):
                    encoded = model.vae.encode(cond).latent_dist.sample()
                    return encoded * model.vae.config.scaling_factor

                sketch_enc = encode_visual_condition(sketch)
                instance_enc = encode_visual_condition(instance)
                depth_enc = encode_visual_condition(depth)
                intensity_enc = encode_visual_condition(intensity)

            # 处理视觉条件（应用dropout）
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

            # print(f"Model input shape: {model_input.shape}")
            # print(f"Timesteps shape: {timesteps.shape}")
            # print(f"Encoder hidden states shape: {encoder_hidden_states.shape}")
            # print(f"Time condition shape: {time_cond.shape}")

            # 预测噪声
            noise_pred = model.unet(
                model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=time_cond
            ).sample

            # print(f"Noise prediction shape: {noise_pred.shape}")
            # print(f"Noise shape: {noise.shape}")

            # 计算损失
            loss = criterion(noise_pred, noise)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    # 保存训练好的模型
    save_path = "TrainedComposer"
    model.save_custom_pretrained(save_path)
    print(f"Model saved to {save_path}")

    return model


# 主函数
if __name__ == "__main__":
    # 初始化模型
    print("Loading model...")
    composer_model = ComposerStableDiffusionPipeline.load_custom_pretrained(
        load_directory=None,  # 加载基础模型
        base_unet_model_id="runwayml/stable-diffusion-v1-5"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # 冻结不需要训练的模块
    for param in composer_model.vae.parameters():
        param.requires_grad = False
    for param in composer_model.text_encoder.parameters():
        param.requires_grad = False
    for param in composer_model.image_encoder.parameters():
        param.requires_grad = False

    print("Model loaded. Starting training...")

    # 开始训练
    trained_model = train_composer_model(
        model=composer_model,
        num_epochs=5,
        batch_size=2,
        learning_rate=1e-5
    )

    print("Training completed!")
