import cv2

# 读取图像
image_path = "results/edge_output.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"图像不存在或路径错误: {image_path}")
else:
    height, width, channels = image.shape
    print(f"图像尺寸（高度, 宽度）: ({height}, {width})")
    print(f"通道数: {channels}")