import os
import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_segmenter(checkpoint_path, model_type="vit_l"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(DEVICE)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92
    )


# def get_segmentation_mask(image_np, mask_generator, output_size=(64,64)):
#     resized_img = cv2.resize(image_np, output_size)
#     masks = mask_generator.generate(resized_img)
#     combined_mask = np.zeros(output_size[::-1], dtype=np.uint8)
#     for m in sorted(masks, key=lambda x: x["area"], reverse=True):
#         combined_mask[m["segmentation"]] = 255
#     return combined_mask

def get_segmentation_mask(image_np, mask_generator, output_size=(512, 512)):
    # 在原图分辨率下生成高质量分割
    masks = mask_generator.generate(image_np)
    masks_sorted = sorted(masks, key=lambda x: x["stability_score"], reverse=True)
    full_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    for m in masks_sorted:
        full_mask[m["segmentation"]] = 255
    resized_mask = cv2.resize(
        full_mask,
        output_size,
        interpolation=cv2.INTER_AREA
    )
    return resized_mask
