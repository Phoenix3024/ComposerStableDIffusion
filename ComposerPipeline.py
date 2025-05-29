import os

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from torch import nn

from ComposerUnet import ComposerUNet


class ComposerStableDiffusionPipeline(StableDiffusionPipeline):
    """
    Custom Stable Diffusion pipeline that integrates a MegaConditionUNet combining CLIP image/text features
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
        - image: PIL Image or tensor to extract CLIP features
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
        uncond_sketch = create_uncond_input(sketch)
        uncond_instance = create_uncond_input(instance)
        uncond_depth = create_uncond_input(depth)
        uncond_intensity = create_uncond_input(intensity)
        uncond_prompt = [""] * batch_size if isinstance(prompt, list) else ""

        # 扩散过程
        for t in self.scheduler.timesteps:
            # 扩展latents用于CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 准备条件输入（原始条件+无条件）
            cond_pixel_values = torch.cat([pixel_values, uncond_pixel_values]) if pixel_values is not None else None
            cond_color = torch.cat([color, uncond_color]) if color is not None else None
            cond_sketch = torch.cat([sketch, uncond_sketch]) if sketch is not None else None
            cond_instance = torch.cat([instance, uncond_instance]) if instance is not None else None
            cond_depth = torch.cat([depth, uncond_depth]) if depth is not None else None
            cond_intensity = torch.cat([intensity, uncond_intensity]) if intensity is not None else None

            # 处理文本提示
            if isinstance(prompt, list):
                cond_prompt = prompt + uncond_prompt
            else:
                cond_prompt = [prompt] * batch_size + [uncond_prompt]

            # 预测噪声
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=None,
                image=cond_pixel_values,
                prompt=cond_prompt,
                color=cond_color,
                sketch=cond_sketch,
                instance=cond_instance,
                depth=cond_depth,
                intensity=cond_intensity,
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
        base_unet = UNet2DConditionModel.from_pretrained(base_unet_model_id, cache_dir='data/pretrain_models',
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
