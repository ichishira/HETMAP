import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import random
from PIL import Image, ImageOps
import warnings
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import cv2

import openslide
import h5py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import maxflow
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
import networkx as nx
from scipy.stats import multivariate_normal


from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import timedcall

import umap
import time

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login


import os
import joblib
from joblib import Parallel, delayed
import openslide


warnings.simplefilter('ignore')






def visualize_categorical_heatmap(
        wsi,
        orig_coords, 
        labels, 
        label2color_list,
        vis_level=None,
        patch_size=(224, 224),
        alpha=0.4,
        verbose=True
    ):
    
    downsample = int(wsi.level_downsamples[vis_level])
    scale = [1/downsample, 1/downsample]
    
    
    top_left = (0, 0)
    bot_right = wsi.level_dimensions[0]
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))
    w, h = region_size
    
    patch_size_orig = (224,224)
    patch_size = np.ceil(np.array(patch_size_orig) * np.array(scale)).astype(int)
    downsample_coords = np.ceil(orig_coords * np.array(scale)).astype(int)
    verbose=True
    if verbose:
        print('\nCreating heatmap for: ')
        print('Top Left: ', top_left, 'Bottom Right: ', bot_right)
        print('Width: {}, Height: {}'.format(w, h))
        print(f'Original Patch Size / Scaled Patch Size: {patch_size_orig} / {patch_size}')
    vis_level = wsi.get_best_level_for_downsample(downsample)
    img = wsi.read_region(top_left, vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    if img.size != region_size:
        img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
    img = np.array(img)
    raw_img = img.copy()
    
    if verbose:
        print('vis_level: ', vis_level)
        print('downsample: ', downsample)
        print('region_size: ', region_size)
        print('total of {} patches'.format(len(downsample_coords)))
    
    for idx in tqdm(range(len(downsample_coords))):
        coord = downsample_coords[idx]
        color = label2color_list[labels[idx]][:3]
        color = tuple([int(color_element*255) for color_element in list(color)])
        img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()
        color_block = (np.ones((img_block.shape[0], img_block.shape[1], 3)) * color).astype(np.uint8)
        blended_block = cv2.addWeighted(color_block, alpha, img_block, 1 - alpha, 0)
        blended_block = np.array(ImageOps.expand(Image.fromarray(blended_block), border=1, fill=(50,50,50)).resize((img_block.shape[1], img_block.shape[0])))
        img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = blended_block
        
    raw_img=Image.fromarray(raw_img)
    img = Image.fromarray(img)

    return raw_img, img




