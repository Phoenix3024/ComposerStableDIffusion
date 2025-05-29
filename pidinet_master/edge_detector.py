# edge_detector.py
import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models import pidinet
from models.convert_pidinet import convert_pidinet
from models.pidinet import pidinet_converted

class EdgeDetector:
    def __init__(self, checkpoint_path="trained_models/table7_pidinet.pth", config='carv4'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 配置参数（必须与训练时一致）
        class Args:
            def __init__(self):
                self.config = config
                self.dil = True  # 启用 CDCM
                self.sa = True  # 启用 CSAM
                self.convert = True

        args = Args()

        # 初始化模型
        self.model = pidinet_converted(args)

        # 加载并处理权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
        converted_state_dict = convert_pidinet(state_dict, args.config)

        # 加载权重
        self.model.load_state_dict(converted_state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_edge(self, image_np):
        """输入：RGB numpy数组 (0-255 uint8)，输出：500x500边缘图"""
        image = Image.fromarray(image_np)
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            edge_map = torch.squeeze(outputs[-1].cpu()).numpy()

        edge_map = (edge_map * 255).astype(np.uint8)
        edge_map = cv2.resize(edge_map, (64, 64))
        return edge_map


def main():
    detector = EdgeDetector()

    # 测试图像路径
    test_img_path = "data/apron_05s.jpg"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(test_img_path)
    if image is None:
        raise FileNotFoundError(f"测试图像不存在: {test_img_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    edge_map = detector.get_edge(image_rgb)
    edge_map = cv2.bitwise_not(edge_map)
    edge_map = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(f"{output_dir}/edge_output.jpg", edge_map)
    print(f"结果已保存至 {output_dir}/edge_output.jpg")


if __name__ == "__main__":
    main()