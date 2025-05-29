import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from diffusers import UNet2DConditionModel
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel


class ComposerUNet(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # CLIP融合模块
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir='data/pretrain_models')
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",
                                                            cache_dir='data/pretrain_models')
        self.clip_image_proj = nn.Sequential(
            nn.Linear(768, 1280),
            nn.SiLU(),
            nn.Linear(1280, 2048)
        )
        self.clip_text_proj = nn.Sequential(
            nn.Linear(768, 512)
        )
        self.color_proj = nn.Sequential(
            nn.Linear(156, 512),
            nn.SiLU(),
            nn.Linear(512, 2048)
        )
        self.clip_image_time_proj = nn.Sequential(
            nn.Conv1d(4, 1, 3, padding=1, stride=1),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 320)
            # nn.Linear(256, 128),
            # nn.SiLU(),
            # nn.Linear(128, 64),
            # nn.SiLU(),
            # nn.Linear(64, 16),
            # nn.SiLU(),
            # nn.Linear(16, 1)
        )
        self.clip_text_time_proj = nn.Sequential(
            nn.Conv1d(77, 64, 3, padding=1, stride=1),
            nn.Conv1d(64, 32, 3, padding=1, stride=1),
            nn.Conv1d(32, 16, 3, padding=1, stride=1),
            nn.Conv1d(16, 1, 3, padding=1, stride=1),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 320)
            # nn.Linear(256, 128),
            # nn.SiLU(),
            # nn.Linear(128, 64),
            # nn.SiLU(),
            # nn.Linear(64, 16),
            # nn.SiLU(),
            # nn.Linear(16, 1)
        )
        self.color_time_proj = nn.Sequential(
            nn.Conv1d(4, 1, 3, padding=1, stride=1),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 320)
            # nn.Linear(256, 128),
            # nn.SiLU(),
            # nn.Linear(128, 64),
            # nn.SiLU(),
            # nn.Linear(64, 16),
            # nn.SiLU(),
            # nn.Linear(16, 1)
        )

        # 视觉条件处理
        self.local_condition_proj = LocalConditionProj()
        self.clip_proj = nn.Sequential(
            nn.Linear(512, 768),
            nn.SiLU(),
            nn.Linear(768, 768)
        )

    def forward(self, sample, timestep, encoder_hidden_states, image,
                prompt, color, sketch, instance, depth, intensity):
        # 第一部分：CLIP融合条件

        # 处理图像和文本
        image_inputs = self.clip_processor(images=image, return_tensors="pt", padding=True, do_rescale=False).to(
            sample.device)
        text_inputs = self.clip_processor(text=prompt, return_tensors="pt", padding="max_length", max_length=77,
                                          truncation=True).to(sample.device)
        # 处理颜色
        color = color.to(sample.device)

        # 提取特征
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**image_inputs)
            text_features = self.clip_model.text_model(**text_inputs).last_hidden_state

            # 处理图像和文本特征
            B = image_features.size(0)
            image_features = self.clip_image_proj(image_features)
            image_features = image_features.view(B, 4, 512)
            text_features = self.clip_text_proj(text_features)
            color_features = self.color_proj(color)
            color_features = color_features.view(B, 4, 512)
            clip_context = torch.cat([image_features, text_features, color_features], dim=1)
            clip_context = self.clip_proj(clip_context)

            # 第二部分：时间步融合

            clip_image_time_emb = self.clip_image_time_proj(image_features).view(B, 320)
            clip_text_time_emb = self.clip_text_time_proj(text_features).view(B, 320)
            color_time_emb = self.color_time_proj(color_features).view(B, 320)
            timestep_cond = clip_image_time_emb + clip_text_time_emb + color_time_emb

            # 第三部分：视觉条件处理

            condition_features = self.local_condition_proj(
                sketch=sketch.to(sample.device),
                instance=instance.to(sample.device),
                depth=depth.to(sample.device),
                intensity=intensity.to(sample.device)
            )
            local_condition = torch.zeros_like(sample)
            for i in range(len(condition_features)):
                local_condition += condition_features[i]

            # 第四部分：输入层融合

            combined_input = torch.cat([sample, local_condition], dim=1)

        # 执行原始UNet前向传播
        return super().forward(combined_input,
                               timestep,
                               encoder_hidden_states=clip_context,
                               timestep_cond=timestep_cond,
                               return_dict=False)


class LocalConditionProj(nn.Module):
    def __init__(self):
        super().__init__()
        # 多条件视觉特征提取
        self.condition_convs = nn.ModuleDict({
            'sketch': nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=1),
                nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=1),
                nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=1)
            ),
            'instance': nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=1),
                nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=1),
                nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=1)
            ),
            'depth': nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=1),
                nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=1),
                nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=1)
            ),
            'intensity': nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=1),
                nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=1),
                nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=1)
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


class ComposerDataset(Dataset):
    def __init__(self,
                 num_samples=123403,
                 unlabeled_dir="unlabeled2017",
                 feature_dir="feature_maps",
                 caption_csv="caption.csv",
                 filenames_npy="filenames.npy",
                 color_npy="color.npy"):
        self.num_samples = num_samples
        self.unlabeled_dir = unlabeled_dir
        self.feature_dir = feature_dir

        # 加载颜色数据
        self.filenames = np.load(filenames_npy)[:num_samples]
        self.colors = torch.from_numpy(np.load(color_npy)[:num_samples]).float()

        # 加载caption数据
        self.caption_df = pd.read_csv(caption_csv)
        self.caption_map = {
            os.path.basename(row.image_path): row.caption
            for _, row in self.caption_df.iterrows()
        }

        # 定义图像变换
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.feature_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # 归一化CLIP参数
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 获取基础文件名
        filename = self.filenames[idx]
        base_name = os.path.splitext(filename)[0]

        # 加载原始图像
        image = Image.open(os.path.join(self.unlabeled_dir, filename)).convert("RGB")

        # print(self.caption_map)

        # 加载特征数据
        def load_feature(feature_type):
            path = os.path.join(
                self.feature_dir,
                feature_type,
                f"{base_name}_{feature_type}.{'png'}"
            )
            return self.feature_transform(Image.open(path).convert("RGB"))

        return {
            "image": self.image_transform(image),
            "pixel_values": self.clip_transform(image),
            "prompt": self.caption_map[filename],
            "color": self.colors[idx],
            "sketch": load_feature("sketch"),
            "instance": load_feature("segment"),
            "depth": load_feature("depth"),
            "intensity": load_feature("intensity")
        }
