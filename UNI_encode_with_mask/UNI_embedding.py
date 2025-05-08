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


def ret_masked_img(hist_img, tum_img, tum_n_img, level):
    tum_img = tum_img.convert("RGB")
    tum_img_np = np.array(tum_img)
    tum_gray_img = cv2.cvtColor(tum_img_np, cv2.COLOR_RGB2GRAY)

    tum_n_img = tum_n_img.convert("RGB")
    tum_n_img_np = np.array(tum_n_img)
    tum_n_gray_img = cv2.cvtColor(tum_n_img_np, cv2.COLOR_RGB2GRAY)

    tum_mask = np.where((tum_gray_img + tum_n_gray_img == 0), 0, 1)

    hist_img = hist_img.convert("RGB")
    hist_img_np = np.array(hist_img)

    masked_np = hist_img_np * tum_mask[:, :, np.newaxis]
    masked_img = Image.fromarray(masked_np.astype(np.uint8))

    if level==40:
        hist_img_resized = hist_img.resize((224,224), resample=Image.NEAREST)
        masked_img_resized = masked_img.resize((224,224), resample=Image.NEAREST)
        return hist_img_resized, masked_img_resized
    elif level==20:
        return hist_img, masked_img
    else:
        print("Invalid level")


class Dataset_for_sample_with_mask(Dataset):
    def __init__(self, wsi, tum_wsi, tum_n_wsi, coords, transform, level, patch_size):
        self.wsi = wsi
        self.tum_wsi = tum_wsi
        self.tum_n_wsi = tum_n_wsi
        self.coords = coords
        self.level = level
        self.num_samples = len(self.coords)
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        patch_coords = self.coords[idx]

        hist_img = self.wsi.read_region((patch_coords[0], patch_coords[1]), 0, (self.patch_size, self.patch_size))
        tum_img = self.tum_wsi.read_region((patch_coords[0], patch_coords[1]), 0, (self.patch_size, self.patch_size))
        tum_n_img = self.tum_n_wsi.read_region((patch_coords[0], patch_coords[1]), 0, (self.patch_size, self.patch_size))

        patch_img, masked_patch_img = ret_masked_img(hist_img, tum_img, tum_n_img, self.level)

        if patch_img is not None:
            patch_img_tensor = self.transform(patch_img)
            masked_patch_img_tensor = self.transform(masked_patch_img)
            return patch_img_tensor, masked_patch_img_tensor, patch_coords
        else:
            return None, patch_coords


# ========== メイン処理 ==========
def main():
    args = parse_args()
    sample_id = args.sample
    cancer_type=args.cancertype
    save_dir = args.savedir
    level = args.level

    print(f"Processing sample: {sample_id}")

    # ========== ファイルパス設定 ==========
    if level ==20:
        slide_path = f"/wsi/analysis/TCGA_TMA_analysis/data/WSI/{cancer_type}/{sample_id}.svs"
        tum_path = f"/wsi/analysis/TCGA_TMA_analysis/data/paget/{cancer_type}/{sample_id}/tum.tif"
        tum_n_path = f"/wsi/analysis/TCGA_TMA_analysis/data/paget/{cancer_type}/{sample_id}/tum_n.tif"
        h5_file_path = f"/data/shirasuna/work/hetero_path/TCGA_wsi_patch/patched_{cancer_type}_mag20/patches/{sample_id}.h5"

    elif level==40:
        slide_path = f"/wsi/analysis/TCGA_TMA_analysis/data/WSI/{cancer_type}/{sample_id}.svs"
        tum_path = f"/wsi/analysis/TCGA_TMA_analysis/data/paget/{cancer_type}/{sample_id}/tum.tif"
        tum_n_path = f"/wsi/analysis/TCGA_TMA_analysis/data/paget/{cancer_type}/{sample_id}/tum_n.tif"
        h5_file_path = f"/data/shirasuna/work/hetero_path/TCGA_wsi_patch/patched_{cancer_type}_mag40/patches/{sample_id}.h5"
        
    hist_wsi = openslide.OpenSlide(slide_path)
    tum_wsi = openslide.OpenSlide(tum_path)
    tum_n_wsi = openslide.OpenSlide(tum_n_path)
    h5_wsi = h5py.File(h5_file_path, 'r')

    wsi_patch_coords = np.array(h5_wsi["coords"])


    # ========== 腫瘍パッチ抽出 ==========
    if level==40:
        patch_size = 448
    elif level==20:
        patch_size = 224
 

    def process_patch(coord):
        tum_img_np = np.asarray(tum_wsi.read_region((coord[0], coord[1]), 0, (patch_size, patch_size)).convert("RGB"))
        tum_gray_img = cv2.cvtColor(tum_img_np, cv2.COLOR_RGB2GRAY)

        tum_n_img_np = np.asarray(tum_n_wsi.read_region((coord[0], coord[1]), 0, (patch_size, patch_size)).convert("RGB"))
        tum_n_gray_img = cv2.cvtColor(tum_n_img_np, cv2.COLOR_RGB2GRAY)

        tum_mask = (tum_gray_img + tum_n_gray_img > 0).astype(np.uint8)

        return coord if tum_mask.sum() > (tum_mask.size / 4) else None

    coords_with_tumor = Parallel(n_jobs=30, backend="threading")(
        delayed(process_patch)(coord) for coord in wsi_patch_coords
    )
    coords_with_tumor = np.stack([c for c in coords_with_tumor if c is not None])

    # ========== データローダー作成 ==========
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = Dataset_for_sample_with_mask(hist_wsi, tum_wsi, tum_n_wsi, coords_with_tumor, transform, level, patch_size)
    test_dataloader = DataLoader(test_dataset, num_workers=8, batch_size=128, shuffle=False)

    # ========== 特徴量抽出 ==========
    model_path = '/home/shirasuna/work/hetero_path/UNI/raw_UNI2-h.pth'
    model = torch.load(model_path)
    model.to('cuda')
    model.eval()

    features_list = []
    coords_list = []

    with torch.no_grad():
        for _, images, coords in test_dataloader:
            images = images.to('cuda')
            features = model(images)
            
            features_list.append(features.cpu())
            coords_list.append(coords.cpu())

    test_features = torch.cat(features_list, dim=0).numpy()
    test_coords = torch.cat(coords_list, dim=0).numpy()

    print(f"patch_number:{test_coords.shape[0]}")
    # ========== 保存 ==========
    SAVE_DIR = f"/data/shirasuna/work/hetero_path/UNI_encode_with_mask/{save_dir}/{sample_id}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(f"{SAVE_DIR}/coords.npy", test_coords)
    np.save(f"{SAVE_DIR}/features.npy", test_features)

    print(f"Features saved in {SAVE_DIR}")


if __name__ == "__main__":
    main()
