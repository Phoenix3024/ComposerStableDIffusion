import os
import sys
import time

# 模块路径设置
sys.path.append("MiDaS-master")
sys.path.append("segment-anything-main")
sys.path.append("pidinet-master")
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path += [
    os.path.join(current_dir, "MiDaS_master"),
    os.path.join(current_dir, "segment_anything_main"),
    os.path.join(current_dir, "pidinet_master"),
    current_dir
]

import cv2
import numpy as np
import torch
from tqdm import tqdm
from MiDaS_master.run import init_depth_model, get_depth_map
from segment_anything_main.test import init_segmenter, get_segmentation_mask
from Intensity import IntensityGenerator
from pidinet_master.edge_detector import EdgeDetector


def to_3channels(x):
    if x.ndim == 2:
        x = np.repeat(x[:, :, np.newaxis], 3, axis=2)
    return x


class FeatureGenerator:
    def __init__(self, device="cuda"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device(device)

        # 初始化深度模型
        depth_weight_path = os.path.join(current_dir, "MiDaS_master/weights/dpt_beit_large_512.pt")
        self.depth_model, self.depth_transform, _ = init_depth_model(model_path=depth_weight_path)
        self.depth_model.to(self.device)

        # 初始化分割模型（自动使用GPU）
        segment_ckpt = os.path.join(current_dir, "segment_anything_main/sam_vit_l_0b3195.pth")
        self.segmenter = init_segmenter(segment_ckpt)

        # 初始化草图模型
        sketch_ckpt = os.path.join(current_dir, "pidinet_master/trained_models/table7_pidinet.pth")
        self.sketch_gen = EdgeDetector(checkpoint_path=sketch_ckpt)
        if hasattr(self.sketch_gen, 'model'):
            self.sketch_gen.model.to(self.device)

        self.intensity_gen = IntensityGenerator()

    def __del__(self):
        """安全释放资源"""
        attrs = ['depth_model', 'segmenter', 'sketch_gen']
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()

    def process_single(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理草图
        with torch.no_grad():
            if hasattr(self.sketch_gen, 'get_edge'):
                sketch_map = self.sketch_gen.get_edge(image_rgb)
            else:
                raise AttributeError("Sketch generator缺少get_edge方法")
        sketch_map = cv2.bitwise_not(sketch_map)

        return {
            "depth": to_3channels(get_depth_map(image_rgb, self.depth_model, self.depth_transform, self.device)),
            "instance": to_3channels(get_segmentation_mask(image_rgb, self.segmenter)),
            "sketch": to_3channels(sketch_map),
            "intensity": to_3channels(self.intensity_gen.get_intensity(image_rgb))
        }

    def process_batch(self, file_list, input_dir, output_dir):
        total_files = len(file_list)
        start_time = time.time()

        with tqdm(total=total_files, desc='Processing', unit='img') as pbar:
            for idx, img_path in enumerate(file_list, 1):
                try:
                    results = self.process_single(img_path)
                    if output_dir:
                        self._save_results(results, img_path, input_dir, output_dir)

                    # 定期清理缓存
                    if idx % 100 == 0:
                        torch.cuda.empty_cache()

                    # 更新进度信息
                    elapsed = time.time() - start_time
                    speed = elapsed / idx
                    remaining = (total_files - idx) * speed

                    pbar.set_postfix({
                        'remaining': f"{remaining:.1f}s",
                        'speed': f"{speed:.2f}s/img"
                    })
                    pbar.update(1)

                except Exception as e:
                    pbar.write(f"处理失败: {img_path} - {str(e)}")
                    pbar.update(1)

    def _save_results(self, results, img_path, input_root, output_root):
        rel_path = os.path.relpath(img_path, input_root)
        base_name = os.path.splitext(rel_path)[0]
        for feature in ['depth', 'instance', 'sketch', 'intensity']:
            output_dir = os.path.join(output_root, feature, os.path.dirname(rel_path))
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, base_name + f'_{feature}.jpg'), results[feature])


if __name__ == "__main__":
    # 参数配置
    input_dir = "data/unlabeled2017"
    output_dir = "data/feature_maps"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取处理范围
    file_list = []
    for root, _, files in sorted(os.walk(input_dir)):
        for file in sorted(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_list.append(os.path.join(root, file))

    print(f"发现 {len(file_list)} 张待处理图像")

    # 初始化处理器
    feature_gen = FeatureGenerator(device="cuda")

    # 执行处理
    feature_gen.process_batch(file_list, input_dir, output_dir)

    # 完成标记
    with open(os.path.join(output_dir, "COMPLETED"), 'w') as f:
        f.write(f"Processed {len(file_list)} images at {time.ctime()}")
