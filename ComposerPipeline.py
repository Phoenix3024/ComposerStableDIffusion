import os

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPModel, CLIPProcessor

from ComposerUnet import ComposerUNet


class LocalConditionProj(nn.Module):
    def __init__(self):
        super().__init__()
        # 多条件视觉特征提取
        self.condition_convs = nn.ModuleDict({
            'sketch': nn.Sequential(
                nn.Identity()
            ),
            'instance': nn.Sequential(
                nn.Identity()
            ),
            'depth': nn.Sequential(
                nn.Identity()
            ),
            'intensity': nn.Sequential(
                nn.Identity()
            )
        })

    def dropout_conditions(self, sketch, instance, depth, intensity):
        """
        对四个条件张量应用自定义的 Dropout 策略，并返回新的张量列表 [sketch', instance', depth', intensity']。
        - sketch、instance、depth 独立以 0.5 概率丢弃
        - intensity 以 0.7 概率丢弃
        - 0.1 概率丢弃所有条件，0.1 概率保留所有条件
        """
        device = sketch.device  # 确保在相同设备上生成随机数
        rand = torch.rand(1, device=device).item()  # 随机判断是否全丢弃或全保留

        if rand < 0.1:
            # 以0.1概率丢弃所有条件
            return [
                torch.zeros_like(sketch),
                torch.zeros_like(instance),
                torch.zeros_like(depth),
                torch.zeros_like(intensity)
            ]
        elif rand < 0.2:
            # 以0.1概率保留所有条件
            return [
                sketch.clone(),
                instance.clone(),
                depth.clone(),
                intensity.clone()
            ]
        else:
            # 其他情况下独立决策是否丢弃每个条件
            drop_sketch = torch.rand(1, device=device).item() < 0.5
            drop_instance = torch.rand(1, device=device).item() < 0.5
            drop_depth = torch.rand(1, device=device).item() < 0.5
            drop_intensity = torch.rand(1, device=device).item() < 0.7

            # 如果标志为 True 则置零，否则返回原张量的克隆
            new_sketch = torch.zeros_like(sketch) if drop_sketch else sketch.clone()
            new_instance = torch.zeros_like(instance) if drop_instance else instance.clone()
            new_depth = torch.zeros_like(depth) if drop_depth else depth.clone()
            new_intensity = torch.zeros_like(intensity) if drop_intensity else intensity.clone()

            return [new_sketch, new_instance, new_depth, new_intensity]

    def forward(self, sketch, instance, depth, intensity):
        # 处理各条件特征
        local_conds = {
            'sketch': sketch,
            'instance': instance,
            'depth': depth,
            'intensity': intensity
        }

        condition_features = []
        for name, conv in self.condition_convs.items():
            feat = conv(local_conds[name])
            condition_features.append(feat)

        new_condition_features = self.dropout_conditions(
            condition_features[0],
            condition_features[1],
            condition_features[2],
            condition_features[3]
        )

        return new_condition_features


class ComposerStableDiffusionPipeline(StableDiffusionPipeline):
    """
    Custom Stable Diffusion pipeline that integrates a ConditionUNet combining CLIP text features
    with additional local conditions (color, sketch, instance, depth, intensity).
    """

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: ComposerUNet,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: None,
            requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
            safety_checker=safety_checker, feature_extractor=feature_extractor, image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker
        )
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
                              safety_checker=safety_checker, feature_extractor=feature_extractor)
        self.logger = logging.get_logger(__name__)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # CLIP融合模块
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir='data/pretrain_models')
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",
                                                            cache_dir='data/pretrain_models')
        self.color_proj = nn.Sequential(
            nn.Linear(156, 512),
            nn.SiLU(),
            nn.Linear(512, 2048),
            nn.SiLU(),
            nn.Linear(2048, 768 * 4)
        )
        self.clip_image_time_proj = nn.Sequential(
            nn.Conv1d(4, 1, 3, padding=1, stride=1),
            nn.Linear(768, 256),
            nn.SiLU(),
            nn.Linear(256, 320)
        )

        self.clip_text_time_proj = nn.Sequential(
            nn.Conv1d(77, 64, 3, padding=1, stride=1),
            nn.Conv1d(64, 32, 3, padding=1, stride=1),
            nn.Conv1d(32, 16, 3, padding=1, stride=1),
            nn.Conv1d(16, 1, 3, padding=1, stride=1),
            nn.Linear(768, 256),
            nn.SiLU(),
            nn.Linear(256, 320)
        )
        self.color_time_proj = nn.Sequential(
            nn.Conv1d(4, 1, 3, padding=1, stride=1),
            nn.Linear(768, 256),
            nn.SiLU(),
            nn.Linear(256, 320)
        )

        # 视觉条件处理
        self.local_condition_proj = LocalConditionProj()

    @torch.no_grad()
    def __call__(
            self,
            image: torch.Tensor or Image.Image,
            pixel_values: torch.Tensor,
            prompt: str,
            color: torch.Tensor = None,
            sketch: torch.Tensor = None,
            instance: torch.Tensor = None,
            depth: torch.Tensor = None,
            intensity: torch.Tensor = None,
            guidance_scale: float = 1.0,
            num_inference_steps: int = 50,
    ):
        """
        Generate an image conditioned on text prompt and additional inputs:
        - prompt: text string
        - color, sketch, instance, depth, intensity: tensors for local conditioning
        """
        # 获取设备信息
        device = self.device

        # 确定batch size
        batch_size = pixel_values.shape[0] if isinstance(pixel_values, torch.Tensor) else 1

        # 准备初始噪声
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, torch.Tensor):
            _, _, height, width = image.shape
        else:
            height = width = 512

        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            device=device,
            dtype=torch.float32,
        )
        latents = latents * self.scheduler.init_noise_sigma

        pixel_values = self.clip_processor(images=pixel_values, return_tensors="pt", padding=True, do_rescale=False).to(
            device)
        pixel_values = self.clip_model.get_image_features(**pixel_values)

        prompt = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(device)
        prompt = self.text_encoder(prompt)[0]
        uncond_prompt = [""] * batch_size if isinstance(prompt, list) else ""
        uncond_prompt = self.tokenizer(
            uncond_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(device)
        uncond_prompt = self.text_encoder(uncond_prompt)[0]
        color = self.color_proj(color)
        sketch = self.vae.encode(sketch).latent_dist.sample()
        sketch = sketch * self.vae.config.scaling_factor
        instance = self.vae.encode(instance).latent_dist.sample()
        instance = instance * self.vae.config.scaling_factor
        depth = self.vae.encode(depth).latent_dist.sample()
        depth = depth * self.vae.config.scaling_factor
        intensity = self.vae.encode(intensity).latent_dist.sample()
        intensity = intensity * self.vae.config.scaling_factor

        # 设置时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # 准备CFG所需的无条件输入
        def create_uncond_input(input_tensor):
            if input_tensor is None:
                return None
            if isinstance(input_tensor, torch.Tensor):
                return torch.zeros_like(input_tensor)
            elif isinstance(input_tensor, list):
                return [""] * len(input_tensor)
            return None

        uncond_pixel_values = create_uncond_input(pixel_values)
        uncond_color = create_uncond_input(color)
        B = batch_size

        # 扩散过程
        for t in self.scheduler.timesteps:
            # 扩展latents用于CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 准备条件输入（原始条件+无条件）
            time_cond_pixel_values = self.clip_image_time_proj(pixel_values).view(B, 320)
            time_cond_prompt = self.clip_text_time_proj(prompt).view(B, 320)
            time_cond_color = self.color_time_proj(color).view(B, 320)
            time_cond_uncond_pixel_values = self.clip_image_time_proj(uncond_pixel_values).view(B, 320)
            time_cond_uncond_prompt = self.clip_text_time_proj(uncond_prompt).view(B, 320)
            time_cond_uncond_color = self.color_time_proj(uncond_color).view(B, 320)
            time_cond = time_cond_pixel_values + time_cond_uncond_pixel_values + time_cond_prompt + time_cond_color + time_cond_uncond_prompt + time_cond_uncond_color

            cond_pixel_values = torch.cat([pixel_values, uncond_pixel_values])
            cond_prompt = torch.cat([prompt, uncond_prompt])
            cond_color = torch.cat([color, uncond_color])
            cond_encoder_hidden_states = torch.cat([cond_pixel_values, cond_prompt, cond_color], dim=1)

            cond_sketch = torch.cat([sketch] * 2)
            cond_instance = torch.cat([instance] * 2)
            cond_depth = torch.cat([depth] * 2)
            cond_intensity = torch.cat([intensity] * 2)

            # 处理视觉条件
            cond_local_conditions = self.local_condition_proj(
                sketch=cond_sketch,
                instance=cond_instance,
                depth=cond_depth,
                intensity=cond_intensity
            )
            local_conditions = torch.zeros_like(latent_model_input)
            for i in range(len(cond_local_conditions)):
                local_conditions += cond_local_conditions[i]

            # 将视觉条件与latents合并
            latent_model_input = torch.cat([latent_model_input, local_conditions], dim=1)

            # 预测噪声
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=cond_encoder_hidden_states,
                timestep_cond=time_cond
            )[0]

            # 应用分类器自由引导(CFG)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 计算前一个噪声样本 x_{t-1}
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 解码latents为图像
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample

        # 后处理
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (image * 255).round().astype("uint8")
        images = [Image.fromarray(img) for img in images]

        return {"images": images}

    def save_custom_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)

        # 保存 diffusers 标准组件
        super().save_pretrained(save_directory)

        # 保存自定义 UNet 的权重（state_dict）
        torch.save(self.unet.state_dict(), os.path.join(save_directory, "composer_unet.pth"))
        print(f"[保存成功] Pipeline 与自定义 UNet 保存到 {save_directory}")

    @classmethod
    def load_custom_pretrained(cls, load_directory: str or None,
                               base_unet_model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        自定义加载函数：加载 Pipeline + 注入 UNet state_dict。
        参数：
            - load_directory: 保存路径（包含 pipe + composer_unet.pth）
            - base_unet_model_id: 用于获取原始 SD1.5 的 unet config
        """

        # 加载其他组件
        if load_directory is None:
            load_directory = base_unet_model_id
        vae = AutoencoderKL.from_pretrained(load_directory, cache_dir='data/pretrain_models', subfolder="vae")
        text_encoder = CLIPTextModel.from_pretrained(load_directory, cache_dir='data/pretrain_models',
                                                     subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(load_directory, cache_dir='data/pretrain_models',
                                                  subfolder="tokenizer")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(load_directory, cache_dir='data/pretrain_models',
                                                                      subfolder="safety_checker")
        feature_extractor = CLIPImageProcessor.from_pretrained(load_directory, cache_dir='data/pretrain_models',
                                                               subfolder="feature_extractor")
        scheduler = DDPMScheduler.from_pretrained(load_directory, cache_dir='data/pretrain_models',
                                                  subfolder="scheduler")

        # 构造并加载自定义 UNet
        base_unet = UNet2DConditionModel.from_pretrained(load_directory, cache_dir='data/pretrain_models',
                                                         subfolder="unet")

        for config in base_unet.config.keys():
            if config == "time_cond_proj_dim":
                base_unet.config[config] = 320
        custom_unet = ComposerUNet(**base_unet.config)
        custom_unet.conv_in = nn.Conv2d(8, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        if load_directory != "runwayml/stable-diffusion-v1-5":
            state_dict_path = os.path.join(load_directory, "composer_unet.pth")
            custom_unet.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

        # 构造 Pipeline
        pipe = cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=custom_unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=None
        )

        print(f"[加载成功] Pipeline 从 {load_directory} 加载完成")
        return pipe
