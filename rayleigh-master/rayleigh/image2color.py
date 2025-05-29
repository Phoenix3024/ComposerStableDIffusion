# import numpy as np
# from image import Image
# from palette import Palette
# from util import histogram_colors_smoothed
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
#
# # 初始化156色系调色板（关键参数设置）
# palette = Palette(
#     num_hues=12,  # 色调数
#     sat_range=4,  # 饱和度范围
#     light_range=4  # 亮度范围
# )
#
#
# # 加载并处理图像
# def extract_color_histogram(image_path):
#     # 加载图像（自动下采样到最长边240像素）
#     img = Image(image_path)
#
#     # 生成平滑直方图（sigma=10）
#     color_hist = histogram_colors_smoothed(
#         lab_array=img.lab_array,
#         palette=palette,
#         sigma=10,  # 与论文参数一致
#         direct=True  # 使用直接平滑方式
#     )
#
#     # 确保输出维度为156
#     assert len(color_hist) == 156, f"Expected 156D, got {len(color_hist)}D"
#     return color_hist
#
# def visualize_color_hist(hist, palette):
#     plt.figure(figsize=(20, 6))
#
#     # 主图区域设置（占90%高度）
#     main_ax = plt.axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]
#
#     # 绘制条形图
#     main_ax.bar(range(len(hist)), hist,
#                 color=palette.hex_list[:len(hist)],
#                 edgecolor='black', width=1)
#
#     # 设置主坐标轴
#     main_ax.set_ylim(0, hist.max() * 1.1)  # 强制Y轴从0开始
#     main_ax.set_xticks([])
#     main_ax.set_ylabel('Color Frequency')
#     main_ax.set_title('156-D Color Histogram Visualization')
#
#     # 添加渐变条（单独坐标轴）
#     gradient_ax = plt.axes([0.1, 0.05, 0.8, 0.05])  # 底部5%高度
#     gradient = np.linspace(0, 1, 256).reshape(1, -1)
#     gradient_ax.imshow(gradient, aspect='auto', cmap='hsv')
#     gradient_ax.set_axis_off()  # 隐藏渐变条坐标轴
#
#     plt.show()
#
# # 在main中添加调用
# if __name__ == "__main__":
#     hist_156 = extract_color_histogram("aircraft_carrier_06s.jpg")
#     print("156维颜色向量：", hist_156)
#
#     # 可视化
#     visualize_color_hist(hist_156, palette)

import os

import numpy as np
from tqdm import tqdm  # 用于显示进度条

from image import Image
from palette import Palette
from util import histogram_colors_smoothed

# 初始化156色系调色板
palette = Palette(
    num_hues=12,
    sat_range=4,
    light_range=4
)


def process_dataset(image_dir, output_dir):
    """
    批量处理图像数据集
    :param image_dir: 输入图像目录路径（包含所有jpg文件）
    :param output_dir: 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有jpg文件路径
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    print(f"发现 {len(image_files)} 张图像需要处理")

    # 预分配存储空间
    all_histograms = np.zeros((len(image_files), 156), dtype=np.float32)
    filenames = []

    # 处理进度条
    progress = tqdm(image_files, desc="处理图像", unit="img")

    # 批量处理
    success_count = 0
    error_log = []

    for idx, filename in enumerate(progress):
        try:
            # 生成颜色直方图
            img_path = os.path.join(image_dir, filename)
            hist = extract_color_histogram(img_path)

            # 存储结果
            all_histograms[idx] = hist
            filenames.append(filename)
            success_count += 1

        except Exception as e:
            error_log.append({
                "filename": filename,
                "error": str(e)
            })
            progress.write(f"错误处理 {filename}: {str(e)}")

    # 保存结果
    np.save(os.path.join(output_dir, "color_histograms.npy"), all_histograms[:success_count])
    np.save(os.path.join(output_dir, "filenames.npy"), filenames)

    # 保存错误日志
    if error_log:
        error_path = os.path.join(output_dir, "error_log.csv")
        with open(error_path, "w") as f:
            f.write("filename,error\n")
            for entry in error_log:
                f.write(f"{entry['filename']},{entry['error']}\n")
        print(f"警告：发现 {len(error_log)} 个错误，已保存到 {error_path}")

    print(f"处理完成！成功处理 {success_count} 张图像")
    print(f"结果保存在 {output_dir} 目录：")
    print("- color_histograms.npy: 颜色直方图数据（形状：{all_histograms.shape}）")
    print("- filenames.npy: 对应的文件名列表")


# 原颜色提取函数保持不变
def extract_color_histogram(image_path):
    img = Image(image_path)
    color_hist = histogram_colors_smoothed(
        lab_array=img.lab_array,
        palette=palette,
        sigma=10,
        direct=True
    )
    assert len(color_hist) == 156
    return color_hist


if __name__ == "__main__":
    # 配置路径
    input_dir = r"data/unlabeled2017"  # 输入图像目录
    output_dir = r"data/color_features"  # 输出目录

    # 开始处理
    process_dataset(image_dir=input_dir, output_dir=output_dir)