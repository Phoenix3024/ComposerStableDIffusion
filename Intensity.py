import cv2
import numpy as np
import random


class IntensityGenerator:
    def __init__(self):
        self.weights_set = [
            [0.299, 0.587, 0.114],  # 传统RGB权重
            [0.2126, 0.7152, 0.0722],  # sRGB标准
            [0.333, 0.333, 0.334]  # 均等权重
        ]

    def get_intensity(self, image_np, output_size=(512, 512)):
        """输入RGB numpy数组 (0-255 uint8), 输出强度图"""
        # 随机选择一组权重
        weights = random.choice(self.weights_set)  # 关键修复：从self.weights_set获取

        # 计算强度
        resized = cv2.resize(image_np, output_size)
        intensity = np.dot(resized[..., :3], weights).astype(np.uint8)
        return intensity
