#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import argparse
from PIL import Image
import warnings
import h5py
import openslide
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from func.hetero_cluster_func import (
    create_knn_graph, 
    simulated_annealing, 
    compute_bic,
    compare_bic, 
    compare_bic_integrate,
    recluster_with_kmeans
)
from func.visualize_func import visualize_categorical_heatmap


variance_pca=0.8
warnings.simplefilter('ignore')


# ========== 引数処理 ==========
def parse_args():
    parser = argparse.ArgumentParser(description="clustering UNI embeddings")
    parser.add_argument("-slidedir", type=str, help="slidedir")
    parser.add_argument("-h5dir", type=str, help="h5dir")
    parser.add_argument("-featuredir", type=str, help="featuredir")
    parser.add_argument("-sample", type=str, help="Sample ID")
    parser.add_argument("-maxcluster", type=int, help="maximum cluster number")
    parser.add_argument("-savedir", type=str, help="save_dir")
    parser.add_argument("-seed", type=int, help="seed")
    
    args = parser.parse_args()

    # -sample が指定されていない場合は終了
    if not args.slidedir:
        print("-slidedir argument is required")
        sys.exit(1)  # エラーメッセージを表示して終了
    elif not args.h5dir:
        print("-h5dir argument is required")
        sys.exit(1)  # エラーメッセージを表示して終了
    elif not args.featuredir:
        print("-featuredir argument is required")
        sys.exit(1)  # エラーメッセージを表示して終了
    elif not args.sample:
        print("-sample argument is required")
        sys.exit(1)  # エラーメッセージを表示して終了
    elif not args.maxcluster:
        print("-maximum cluster number is required")
        sys.exit(1)  # エラーメッセージを表示して終了
    elif not args.seed:
        print("-seed is required")
        sys.exit(1)  # エラーメッセージを表示して終了

    return args

# --------------------------
# データ読み込み関数
# --------------------------
def load_data(sample_id, slidedir, h5dir, featuredir):
    """データの読み込みと前処理"""
    print("Loading data")

    slide_path = f"{slidedir}/{sample_id}.svs"
    if not os.path.exists(slide_path):
        slide_path = f"{slidedir}/{sample_id}.ndpi"
    h5_file_path = f"{h5dir}/{sample_id}.h5"
    epi_dir = f"{featuredir}/{sample_id}"

    hist_wsi = openslide.OpenSlide(slide_path)
    h5_wsi = h5py.File(h5_file_path, 'r')
    epi_coords = np.load(f"{epi_dir}/coords.npy")
    epi_features = np.load(f"{epi_dir}/features.npy")

    # 正規化
    epi_features = normalize(epi_features, norm='l2')

    return hist_wsi, h5_wsi, epi_coords, epi_features


# --------------------------
# 次元削減関数
# --------------------------



def reduce_dimensions(epi_features, variance_threshold=0.8, seed=314):
    """分散の累積説明率が指定された閾値（デフォルトは80%）になるようにPCAで次元削減"""
    print("Reducing dimensions using PCA...")
       
    # PCAで次元削減
    scaler = StandardScaler()
    scaled_epi_features= scaler.fit_transform(epi_features)
        
    pca = PCA(random_state=seed)  # 2次元に削減
    embedding = pca.fit_transform(scaled_epi_features)
       
    # 累積分散説明率を計算
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
       
    # 80%以上の分散を説明する次元数を決定
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
       
    # 目標の次元数でPCAを再実行
    pca = PCA(n_components=n_components, random_state=seed)
    embedding = pca.fit_transform(epi_features)
       
    return embedding[:,:n_components]



# def reduce_dimensions(epi_features, variance_threshold=0.8, seed=314):
#     """分散の累積説明率が指定された閾値（デフォルトは80%）になるようにPCAで次元削減"""
#     print("Reducing dimensions using PCA...")
    
#     # PCAで次元削減
#     scaler = StandardScaler()
#     scaled_epi_features= scaler.fit_transform(epi_features)
        
#     pca = PCA(random_state=seed)  # 2次元に削減
#     embedding = pca.fit_transform(scaled_epi_features)

#     return embedding[:,:100]


# --------------------------
# クラスタリング関数
# --------------------------

def cluster(epi_coords, epi_features, embedding, q=10, seed=314):
    """クラスタリング処理"""
    np.random.seed(seed)

    # k-NNグラフ作成
    print("Creating k-NN graph...")
    g, nodes, graph = create_knn_graph(epi_coords, epi_features, k=10)

    # k-meansで初期クラスタリング
    print("Initial k-means clustering...")
    kmeans = KMeans(n_clusters=q, n_init=q, random_state=seed)
    state = kmeans.fit_predict(epi_features)
    cluster_center = kmeans.cluster_centers_

    pre_bic = float('inf')

    # シミュレーテッドアニーリングによる最適化
    print("Starting simulated annealing...")
    finish_flg = 1
    energy_list = []

    while finish_flg != 0:
        finish_flg = 0
        optimized_state, energy_list = simulated_annealing(
            graph, state, cluster_center, epi_features,
            max_iter=10000000, min_iter=10000, 
            T_init=2, T_min=0.01, cooling_rate=0.9
        )

        new_optimized_labels = optimized_state.copy()

        temp_bic = compute_bic(embedding, new_optimized_labels)
        if temp_bic < pre_bic:
            for cluster_id in np.unique(new_optimized_labels):
                
                bic_with, bic_without, labels_without_cluster = compare_bic_integrate(embedding, new_optimized_labels, cluster_id)
        
                # モデル選択
                print(pre_bic)
                if bic_with > bic_without and pre_bic > bic_without: ###clusterの数が減ると0じゃなくなる
                    new_optimized_labels = labels_without_cluster
                    pre_bic = bic_with
                    finish_flg +=1
                    break
        
            if finish_flg!=0:
                state, new_kmeans = recluster_with_kmeans(epi_features, new_optimized_labels)
                cluster_center = new_kmeans.cluster_centers_
                pre_optimized_state = optimized_state.copy()
            
            else:
                if pre_bic<bic_with: ###clusterが一個多い方がbicが低ければそちらにlabelを変更する
                    optimized_state = pre_optimized_state
        else:
            optimized_state = pre_optimized_state

    print("Clustering finished!")
    
    return state, optimized_state


# --------------------------
# 可視化関数
# --------------------------
def visualize(hist_wsi, epi_coords, state, optimized_state, sample_id):
    """クラスタリング結果を可視化"""
    cmap = plt.cm.Spectral
    cluster_num = max(len(np.unique(state)), len(np.unique(optimized_state)))
    cluster_colors = [cmap(i / cluster_num) for i in range(cluster_num)]

    vis_level = hist_wsi.get_best_level_for_downsample(128)

    raw_img, raw_cluster_img = visualize_categorical_heatmap(
        hist_wsi, epi_coords, state, 
        cluster_colors, vis_level=vis_level, 
        patch_size=(224, 224), alpha=1, verbose=True
    )
    
    _, optimized_img = visualize_categorical_heatmap(
        hist_wsi, epi_coords, optimized_state, 
        cluster_colors, vis_level=vis_level, 
        patch_size=(224, 224), alpha=1, verbose=True
    )

    return raw_img, raw_cluster_img, optimized_img



# --------------------------
# メイン関数
# --------------------------
def main():
    seed = 314
    args = parse_args()
    slide_dir = args.slidedir
    h5_dir = args.h5dir
    feature_dir = args.featuredir
    sample_id = args.sample
    maxcluster_num = args.maxcluster
    savedir = args.savedir
    seed = args.seed
    
    # データ読み込み
    hist_wsi, h5_wsi, epi_coords, epi_features = load_data(sample_id, slidedir = slide_dir, h5dir=h5_dir, featuredir=feature_dir)

    # 次元削減
    embedding = reduce_dimensions(epi_features, variance_threshold=variance_pca, seed=seed)

    # クラスタリング
    state, optimized_state = cluster(epi_coords, epi_features, embedding,q=maxcluster_num, seed=seed)

    # 可視化
    raw_img, raw_cluster_img, optimized_img = visualize(hist_wsi, epi_coords, state, optimized_state, sample_id)

    
    #結果の保存
    output_dir = f"/data/shirasuna/work/hetero_path/hetero_cluster/{savedir}/{sample_id}"
    os.makedirs(output_dir, exist_ok=True)

    # clusterの保存
    np.save(f"{output_dir}/raw_state.npy",state)
    np.save(f"{output_dir}/optimized_state.npy", optimized_state)
    
    # 画像の保存
    raw_img_path = os.path.join(output_dir, "he.png")
    raw_cluster_path = os.path.join(output_dir,"clustering_raw.png")
    optimized_cluster_path = os.path.join(output_dir, "clustering_optimized.png")

    raw_img.save(raw_img_path)
    raw_cluster_img.save(raw_cluster_path)
    optimized_img.save(optimized_cluster_path)

    print("Figure saved!")


# --------------------------
# 実行
# --------------------------
if __name__ == "__main__":
    main()
