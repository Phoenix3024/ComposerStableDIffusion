import numpy as np

# 加载数据
histograms = np.load("color_histograms.npy")
filenames = np.load("filenames.npy")

# 查看基本信息
print("颜色直方图数据形状:", histograms.shape)  # 例如 (1000, 256, 3)
print("文件名数组形状:", filenames.shape)    # 例如 (1000,)


# 检查第 i 个样本的对应关系
i = 0  # 任意索引
print("文件名:", filenames[i])
print("对应的直方图数据:", histograms[i])