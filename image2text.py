import csv
import os

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 初始化BLIP模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

# 输入输出路径
image_dir = "data/unlabeled2017"
output_csv = "data/image_captions.csv"

# 创建CSV文件并写入表头
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_id', 'image_path', 'caption'])

    # 遍历图像目录
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)

            try:
                # 生成描述
                image = Image.open(image_path).convert('RGB')
                inputs = processor(image, return_tensors="pt")
                inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
                output = model.generate(**inputs)
                caption = processor.decode(output[0], skip_special_tokens=True)

                # 写入CSV
                writer.writerow([filename, image_path, caption])
                print(f"Processed: {filename} -> {caption[:50]}...")  # 打印进度

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                writer.writerow([filename, image_path, "ERROR"])

print(f"完成！结果已保存到 {output_csv}")