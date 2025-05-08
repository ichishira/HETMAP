#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import random
from PIL import Image
import warnings
import h5py
import openslide
import os
import cv2
from joblib import Parallel, delayed
import time
import argparse
import sys


# Pillowの制限を解除
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore')


# ========== 引数処理 ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Extract UNI embeddings from WSI data")
    parser.add_argument("-sample", type=str, help="Sample ID")
    parser.add_argument("-cancertype", type=str, help="cancer_type")
    parser.add_argument("-savedir", type=str, help="save dir")
    parser.add_argument("-level", type=int, help="magnitude level to encode")
    
    args = parser.parse_args()

    # -sample が指定されていない場合は終了
    if not args.sample:
        print("-sample argument is required")
        print("Usage: ipython UNI_embedding.py -sample sample_id")
        sys.exit(1)  # エラーメッセージを表示して終了

    elif not args.cancertype:
        print("-cancertype argument is required")
        print("Usage: ipython UNI_embedding.py -cancertype cancer_type")
        sys.exit(1)  # エラーメッセージを表示して終了
        
        
    elif not args.savedir:
        print("-savedir argument is required")
        print("Usage: ipython UNI_embedding.py -savedir save_dir")
        sys.exit(1)  # エラーメッセージを表示して終了

    elif not args.level:
        print("-level argument is required")
        print("Usage: ipython UNI_embedding.py -level magnitude_level")
        sys.exit(1)  # エラーメッセージを表示して終了

    return args



class Dataset_for_sample(Dataset):
    def __init__(self, wsi, coords, transform, level, patch_size):
        self.wsi = wsi
        self.coords = coords
        self.level = level
        self.num_samples = len(self.coords)
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        patch_coords = self.coords[idx]

        hist_img = self.wsi.read_region((patch_coords[0], patch_coords[1]), 0, (self.patch_size, self.patch_size)).convert("RGB")
        hist_img_np = np.array(hist_img)
        hist_img=Image.fromarray(hist_img_np.astype(np.uint8))

        if self.level==40:
            patch_img = hist_img.resize((224,224), resample=Image.NEAREST)

        elif self.level==20:
            patch_img = hist_img

        if patch_img is not None:
            patch_img_tensor = self.transform(patch_img)
            return patch_img_tensor, patch_coords
        else:
            return None, patch_coords


# ========== メイン処理 ==========
def main():
    args = parse_args()
    sample_id = args.sample
    cancer_type = args.cancertype
    save_dir = args.savedir
    level = args.level

    print(f"Processing sample: {sample_id}")

    # ========== ファイルパス設定 ==========
    if level ==20:
        slide_path = f"/wsi/analysis/TCGA_TMA_analysis/data/WSI/{cancer_type}/{sample_id}.svs"
        h5_file_path = f"/data/shirasuna/work/hetero_path/TCGA_wsi_patch/patched_{cancer_type}_mag20/patches/{sample_id}.h5"

    elif level==40:
        slide_path = f"/wsi/analysis/TCGA_TMA_analysis/data/WSI/STAD/{sample_id}.svs"
        h5_file_path = f"/data/shirasuna/work/hetero_path/TCGA_wsi_patch/patched_{cancer_type}_mag40/patches/{sample_id}.h5"
        
    hist_wsi = openslide.OpenSlide(slide_path)
    h5_wsi = h5py.File(h5_file_path, 'r')

    wsi_patch_coords = np.array(h5_wsi["coords"])


    # ========== 腫瘍パッチ抽出 ==========
    if level==40:
        patch_size = 448
    elif level==20:
        patch_size = 224
 

    # ========== データローダー作成 ==========
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = Dataset_for_sample(hist_wsi, wsi_patch_coords, transform, level, patch_size)
    test_dataloader = DataLoader(test_dataset, num_workers=15, batch_size=150, shuffle=False)

    # ========== 特徴量抽出 ==========
    model_path = '/home/shirasuna/work/hetero_path/UNI/raw_UNI2-h.pth'
    model = torch.load(model_path)
    model.to('cuda')
    model.eval()

    features_list = []
    coords_list = []

    with torch.no_grad():
        for images, coords in test_dataloader:
            images = images.to('cuda')
            features = model(images)
            
            features_list.append(features.cpu())
            coords_list.append(coords.cpu())

    test_features = torch.cat(features_list, dim=0).numpy()
    test_coords = torch.cat(coords_list, dim=0).numpy()

    print(f"patch_number:{test_coords.shape[0]}")
    # ========== 保存 ==========
    SAVE_DIR = f"/data/shirasuna/work/hetero_path/UNI_encode_wo_mask/{save_dir}/{sample_id}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(f"{SAVE_DIR}/coords.npy", test_coords)
    np.save(f"{SAVE_DIR}/features.npy", test_features)

    print(f"Features saved in {SAVE_DIR}")


if __name__ == "__main__":
    main()
