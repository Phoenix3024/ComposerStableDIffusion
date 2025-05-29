# 0.环境配置

```shell
cd ComposerStableDiffusion-master
conda create -n composer python=3.12
conda activate composer
pip install -r requirements.txt
```

# 1.图像预处理

本项目在COCO数据集上预训练，选用unlabeled2017

## 1.1首先下载COCO数据集并解压

```shell
mkdir data
wget http://images.cocodataset.org/zips/unlabeled2017.zip
unzip -d data/ unlabeled2017.zip
```

## 1.2计算颜色直方图

计算好颜色直方图后，保存到data文件夹下

```shell
python rayleigh-master/rayleigh/image2color.py
```

## 1.3获取图片对应的文字描述

将图片对应的文字描述存储在csv文件中，保存到data文件夹下

```shell
python image2text.py
```

## 1.4计算局部条件（草图、深度图、实例分割图、强度图）

首先下载对应模型的预训练权重（MiDaS、segment-anything）并复制到对应目录下，然后运行预处理脚本

```shell
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
cp dpt_beit_large_512.pt MiDaS_master/weights/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
cp sam_vit_l_0b3195.pth segment_anything_main/
python preprocess.py
```

# 2.训练

## 2.1单卡训练

运行train.py开始训练

```shell
python train.py
```

## 2.2多卡训练

运行train_multi.py进行多卡训练

```python
python train_multi.py
```



# 3.推理

运行infer.py进行推理

```shell
python infer.py
```

