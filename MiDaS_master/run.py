"""Compute depth maps for images (优化版)"""
import os
import torch
import cv2
import numpy as np
from midas.model_loader import default_models, load_model
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image


def init_depth_model(model_type="dpt_beit_large_512", model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_path or os.path.join("weights", f"{model_type}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    transform = Compose([
        Resize(512),  # 在PIL图像上调整尺寸
        ToTensor(),  # 转换PIL Image → Tensor
        Normalize(  # 在Tensor上做归一化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    model, _, _, _ = load_model(device, model_path, model_type, False, None, False)
    return model, transform, device


def get_depth_map(image_np, model, transform, device, output_size=(512, 512)):
    """兼容OpenCV输入的深度图生成"""
    # 转换通道顺序 BGR→RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # 转换为PIL Image (兼容torchvision变换)
    image_pil = Image.fromarray(image_rgb)

    # 应用预处理变换链
    input_tensor = transform(image_pil).unsqueeze(0).to(device)  # [1,3,H,W]

    # 推理
    with torch.no_grad():
        prediction = model(input_tensor)

    # 后处理
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=output_size,
        mode="bilinear",
        align_corners=False
    ).squeeze().cpu().numpy()

    # 归一化到0-255
    return cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# """Compute depth maps for images (修改整合版)"""
# import os
# import torch
# import cv2
# import numpy as np
# from midas.model_loader import default_models, load_model

# def init_depth_model(model_type="dpt_beit_large_512", model_path=None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # model_path = model_path or default_models[model_type]
#     # 指定绝对路径
#     model_path = model_path or os.path.join(
#         os.path.dirname(__file__),  # run.py所在目录（MiDaS_master）
#         "weights", 
#         f"{model_type}.pt"
#     )
#     model, transform, _, _ = load_model(device, model_path, model_type, False, None, False)
#     return model, transform, device

# def get_depth_map(image_np, model, transform, device, output_size=(64,64)):
#     """输入：RGB numpy数组 (0-255 uint8) 输出：500x500灰度图"""
#     # 预处理
#     original_image = image_np.astype(np.float32) / 255.0
#     input_tensor = transform({"image": original_image})["image"]
#     input_tensor = torch.from_numpy(input_tensor).to(device).unsqueeze(0)

#     # 推理
#     with torch.no_grad():
#         prediction = model(input_tensor)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=output_size[::-1],
#             mode="bicubic",
#             align_corners=False
#         ).squeeze().cpu().numpy()

#     # 归一化
#     return cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
