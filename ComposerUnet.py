import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
from PIL import Image
from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from torch.utils.data import Dataset
from torchvision import transforms


class ComposerUNet(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        # 执行原始UNet前向传播
        return super().forward(sample=sample,
                               timestep=timestep,
                               encoder_hidden_states=encoder_hidden_states,
                               timestep_cond=timestep_cond,
                               return_dict=False)


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
            transforms.Resize((512, 512)),
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
                f"{base_name}_{feature_type}.{'jpg'}"
            )
            return self.feature_transform(Image.open(path).convert("RGB"))

        return {
            "image": self.image_transform(image),
            "pixel_values": self.clip_transform(image),
            "prompt": self.caption_map[filename],
            "color": self.colors[idx],
            "sketch": load_feature("sketch"),
            "instance": load_feature("instance"),
            "depth": load_feature("depth"),
            "intensity": load_feature("intensity")
        }
