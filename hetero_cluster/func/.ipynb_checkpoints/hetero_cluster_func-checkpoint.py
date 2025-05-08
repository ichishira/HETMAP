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
from sklearn.metrics import silhouette_score
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

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import os
import joblib
from joblib import Parallel, delayed
import openslide
from scipy.spatial.distance import cdist


warnings.simplefilter('ignore')



def create_knn_graph(coords, features, k=4):
    """
    k-NNグラフを構築
    - coords: xy座標 (N, 2)
    - features: 各点の特徴量 (N, F) でFは特徴量の次元数
    - k: 近傍数
    """
    n = len(coords)
    
    # k-NNモデルの作成 (xy座標に基づいて近傍を計算)
    neighbors = NearestNeighbors(n_neighbors=k + 1)  # 自分自身を含むため +1
    neighbors.fit(coords)
    
    # k-NNの距離とインデックスを計算
    distances, indices = neighbors.kneighbors(coords)

    # maxflowグラフの作成
    g = maxflow.Graph[int]()
    nodes = g.add_nodes(n)

    # NetworkXでグラフ可視化用
    nx_graph = nx.Graph()

    # featuresを使ってコサイン類似度を効率的に計算
    # 全ての特徴量ベクトル間のコサイン類似度を一度に計算
    similarity_matrix = cosine_similarity(features)

    # k-NNでエッジを追加
    for i in range(n):
        for j in range(1, k + 1):  # 自分自身は除外するため1からスタート
            neighbor_idx = indices[i, j]

            # エッジの重み (featureの相関係数) - 事前に計算した相関行列を使用　cos simが高いほど，weight==異なるclusterになった場合のpenalty は小さい
            weight = similarity_matrix[i, neighbor_idx]

            # pymaxflowのグラフにエッジを追加
            g.add_edge(nodes[i], nodes[neighbor_idx], weight, weight)

            # NetworkXのグラフにも追加
            nx_graph.add_edge(i, neighbor_idx, weight=weight)

    return g, nodes, nx_graph







def simulated_annealing(graph, old_state, cluster_center, features, max_iter=1000000,min_iter=10000, T_init=1.5, T_min=0.2, cooling_rate=0.9):
    """
    シミュレーテッドアニーリングによるPottsモデルのエネルギー最小化
    - graph: NetworkXのグラフ（エッジ重み付き）
    - q: スピン状態の数
    - max_iter: 最大イテレーション数
    - T_init: 初期温度
    - T_min: 最小温度
    - alpha: 温度減少率
    """
    nodes_list = list(graph.nodes)
    
    energy_list = []
    low_temperature_energy_list = []
    state = old_state.copy()

    cluster_node_sim = cosine_similarity(cluster_center, features)
    coef_for_energy = len(graph.edges)/len(graph.nodes)  ###nodeによるenergyとedgeによるenergyをバランスさせる
    
    def energy(graph,cluster_node_sim, state):
        """
        Pottsモデルのエネルギー計算
        """
        E = 0
        for u, v, data in graph.edges(data=True):
            if state[u] == state[v]:
                E -= data['weight']  # スピンが異なればエネルギー加算

        for node in graph.nodes: ###
            E -=coef_for_energy*cluster_node_sim[state[node], node]
        
        return E
        

    def calc_delta_energy(graph, cluster_node_sim, state, node, new_spin):
        """
        ノードnodeのスピン変更時のエネルギー差分を計算
        """
        old_spin = state[node]
        if old_spin == new_spin:
            return 0

        delta_E = coef_for_energy*(cluster_node_sim[old_spin, node]-cluster_node_sim[new_spin, node])
    
        for neighbor in graph.neighbors(node):
            weight = graph[node][neighbor]['weight']
            
            # エネルギー差分の計算
            if state[neighbor] == old_spin:
                delta_E += weight  # 古いスピンと同じ → 一致が消えるのでエネルギー上昇
            if state[neighbor] == new_spin:
                delta_E -= weight  # 新しいスピンと一致 → エネルギー減少
            
        return delta_E


    
    # 初期エネルギーの計算
    E = energy(graph,cluster_node_sim, state)
    
    T = T_init

    # 更新するstateの候補
    candidate_nodes=np.unique(state)

    print("simulated annealing start!")
    for epoch in range(1,max_iter):
        # ランダムにノードを選択
        selected_node = random.choice(nodes_list)
        
        # スピンの変更を試みる
        current_spin = state[selected_node]
        new_spin = np.random.choice(candidate_nodes)
        
        if current_spin != new_spin:
            
            # エネルギー差
            delta_E = calc_delta_energy(graph, cluster_node_sim, state, selected_node, new_spin)
            
            # エネルギーが減少した場合は受け入れ
            if delta_E < 0:
                E = E+delta_E
                state[selected_node] = new_spin
            else:
                # 温度が高ければエネルギー増加も受け入れる
                acceptance_prob = np.exp(-delta_E / T)
                if random.random() < acceptance_prob:
                    E = E+delta_E
                    state[selected_node] = new_spin
        
        energy_list.append(E)

        # 温度の減少
        if epoch % 10000 == 0 and T>=T_min:
            T = T*cooling_rate
            
        # 収束条件：エネルギーの変化が小さくなったら終了
        if T<T_min:
            low_temperature_energy_list.append(E)
            if epoch>min_iter and len(low_temperature_energy_list)>100000:
                if abs(E-low_temperature_energy_list[-100000])<abs(low_temperature_energy_list[100000]-low_temperature_energy_list[0])/100:
                   break
        
        
    print(f"optimization_finished epoch:{epoch}")
    
    return state, energy_list



def create_knn_graph_in_feature_space(features, k=1):
    """
    k-NNグラフを構築
    - coords: xy座標 (N, 2)
    - features: 各点の特徴量 (N, F) でFは特徴量の次元数
    - k: 近傍数
    """
    n = len(features)
    
    # k-NNモデルの作成 (xy座標に基づいて近傍を計算)
    neighbors = NearestNeighbors(n_neighbors=k + 1)  # 自分自身を含むため +1
    neighbors.fit(features)
    
    # k-NNの距離とインデックスを計算
    distances, indices = neighbors.kneighbors(features)

    # maxflowグラフの作成
    g = maxflow.Graph[int]()
    nodes = g.add_nodes(n)

    # NetworkXでグラフ可視化用
    nx_graph = nx.Graph()



    # k-NNでエッジを追加
    for i in range(n):
        for j in range(1, k + 1):  # 自分自身は除外するため1からスタート
            neighbor_idx = indices[i, j]

            # エッジの重み (featureの相関係数) - 事前に計算した相関行列を使用　cos simが高いほど，weight==異なるclusterになった場合のpenalty は小さい
            weight = 1

            # pymaxflowのグラフにエッジを追加
            g.add_edge(nodes[i], nodes[neighbor_idx], weight, weight)

            # NetworkXのグラフにも追加
            nx_graph.add_edge(i, neighbor_idx, weight=weight)

    return nx_graph



def compute_bic(X, labels):
    """
    既存のクラスタラベルを基に、各クラスタの分布が正規分布に従うとしてBICを計算する関数
    - X: 特徴量データ (N, D)
    - labels: 既存のクラスタラベル (N,)
    """
    unique_labels = np.unique(labels)
    n_components = len(unique_labels)
    n_samples, n_features = X.shape

    log_likelihood = 0
    parameters = 0
    
    for label in unique_labels:
        cluster_data = X[labels == label]
        n_cluster_samples = len(cluster_data)

        if n_cluster_samples > 0:
            mean = np.mean(cluster_data, axis=0)
            covariance = np.cov(cluster_data.T)

            # 共分散行列が正定値であることを保証するための微小な値を加える
            min_eig = np.linalg.eigvalsh(covariance).min()
            if min_eig < 0:
                covariance += (-min_eig * np.eye(n_features)) + 1e-6 * np.eye(n_features)

            log_likelihood += np.sum(multivariate_normal.logpdf(cluster_data, mean=mean, cov=covariance, allow_singular=True))

            # パラメータ数: 平均 (D) + 共分散行列の独立な要素数 (D*(D+1)/2)
            parameters += n_features + (n_features * (n_features + 1) // 2)

    # クラスタの重みに関するパラメータ数 (n_components - 1) を加える
    parameters += n_components - 1

    bic = -2 * log_likelihood + parameters * np.log(n_samples)

    return bic



def compute_aic(X, labels):
    """
    既存のクラスタラベルを基に、各クラスタの分布が正規分布に従うとしてAICを計算する関数
    - X: 特徴量データ (N, D)
    - labels: 既存のクラスタラベル (N,)
    """
    unique_labels = np.unique(labels)
    n_components = len(unique_labels)
    n_samples, n_features = X.shape

    log_likelihood = 0
    parameters = 0
    
    for label in unique_labels:
        cluster_data = X[labels == label]
        n_cluster_samples = len(cluster_data)

        if n_cluster_samples > 0:
            mean = np.mean(cluster_data, axis=0)
            covariance = np.cov(cluster_data.T)

            # 共分散行列が正定値であることを保証するための微小な値を加える
            min_eig = np.linalg.eigvalsh(covariance).min()
            if min_eig < 0:
                covariance += (-min_eig * np.eye(n_features)) + 1e-6 * np.eye(n_features)

            log_likelihood += np.sum(multivariate_normal.logpdf(cluster_data, mean=mean, cov=covariance, allow_singular=True))

            # パラメータ数: 平均 (D) + 共分散行列の独立な要素数 (D*(D+1)/2)
            parameters += n_features + (n_features * (n_features + 1) // 2)

    # クラスタの重みに関するパラメータ数 (n_components - 1) を加える
    parameters += n_components - 1

    aic = -2 * log_likelihood + parameters * 2

    return aic

# クラスタありとなしのデータでAICを計算
def compare_aic(embedding, optimized_labels, remove_cluster_id):


    # クラスタなしデータ（特定クラスタを除外し，最近傍のnodeのclassで置換する）
    unique_cluster = np.unique(optimized_labels)
    feture_graph = create_knn_graph_in_feature_space(embedding, k=5)
    labels_without_cluster = optimized_labels.copy()
    for node in range(len(optimized_labels)):
        if labels_without_cluster[node] == remove_cluster_id:
            ###remove_cluster_id以外からsampling
            candidate_cluster = np.unique(labels_without_cluster)
            candidate_cluster = candidate_cluster[candidate_cluster != remove_cluster_id]
            new_cluster_id = random.sample(list(candidate_cluster),1)[0]
            ###最近傍のnodeのclusterに変更
            for neighbor_node in feture_graph.neighbors(node):
                if optimized_labels[neighbor_node]!=remove_cluster_id:
                    new_cluster_id = optimized_labels[neighbor_node]
                    break
            labels_without_cluster[node] = new_cluster_id
        
    # AIC計算
    aic_with = compute_aic(embedding, optimized_labels)
    aic_without = compute_aic(embedding, labels_without_cluster)

    # 結果表示
    print(f"AIC with cluster {remove_cluster_id}: {aic_with}  AIC without cluster {remove_cluster_id}: {aic_without}")

    return aic_with, aic_without, labels_without_cluster


# クラスタありとなしのデータでAICを計算(再配置するclusterは最近傍の重心を持つcluster
def compare_aic_grav(embedding, optimized_labels, remove_cluster_id):

    n_clusters = len(np.unique(optimized_labels))
    unique_labels = np.unique(optimized_labels)
    labels_without_cluster = optimized_labels.copy()
    
    cluster_centers = np.array([
        embedding[optimized_labels == cluster_id].mean(axis=0) for cluster_id in unique_labels
    ])  # 各クラスタの重心を計算

    # 除外対象のクラスタのデータ点を抽出
    mask_remove = (optimized_labels == remove_cluster_id)
    mask_keep = ~mask_remove
    

    # 除外対象データ点の座標
    remove_points = embedding[mask_remove]

    for node in range(len(labels_without_cluster)):
        temp_cluster =labels_without_cluster[node]
        if temp_cluster == remove_cluster_id:
            new_label = -1
            new_dist = float('inf')
            for cand_cluster in np.unique(optimized_labels):
                if cand_cluster != remove_cluster_id:
                    temp_distance  = np.linalg.norm(cluster_centers[cand_cluster,:] - embedding[node,:])
                    if temp_distance<new_dist:
                        new_label=cand_cluster
                        new_dist=temp_distance
            labels_without_cluster[node]=new_label
            
        


    # AIC計算
    aic_with = compute_aic(embedding, optimized_labels)
    aic_without = compute_aic(embedding, labels_without_cluster)

    # 結果表示
    print(f"AIC with cluster {remove_cluster_id}: {aic_with}  AIC without cluster {remove_cluster_id}: {aic_without}")

    return aic_with, aic_without, labels_without_cluster


def compare_bic_integrate(embedding, optimized_labels, remove_cluster_id):


    # クラスタなしデータ（特定クラスタを除外し，最近傍のnodeのclassで置換する）
    unique_cluster = np.unique(optimized_labels)
    labels_without_cluster = optimized_labels.copy()

    cluster_centers = np.array([
        embedding[optimized_labels == cluster_id].mean(axis=0) for cluster_id in unique_cluster
    ])  # 各クラスタの重心を計算

    
    num_clusters = cluster_centers.shape[0]
    center_removed = cluster_centers[remove_cluster_id].reshape(1, -1)
    
    # 対象クラスタ以外のインデックス
    other_indices = [i for i in range(num_clusters) if i != remove_cluster_id]
    other_centers = cluster_centers[other_indices]
    
    # 距離を計算
    distances = cdist(center_removed, other_centers)[0]
    
    # 最も近いクラスタ（インデックスは other_indices から逆引き）
    min_index = np.argmin(distances)
    closest_cluster = other_indices[min_index]
    cluster_for_removed_cluster = closest_cluster
    
    for node in range(len(optimized_labels)):
        if labels_without_cluster[node] == remove_cluster_id:
            labels_without_cluster[node] = cluster_for_removed_cluster
        
    # BIC計算
    print(f"{np.unique(optimized_labels)}, {np.unique(labels_without_cluster)}")
    bic_with = compute_bic(embedding, optimized_labels)
    bic_without = compute_bic(embedding, labels_without_cluster)

    # 結果表示
    print(f"BIC with cluster {remove_cluster_id}: {bic_with}  BIC without cluster {remove_cluster_id}: {bic_without}")

    return bic_with, bic_without, labels_without_cluster



# クラスタありとなしのデータでBICを計算
def compare_bic(embedding, optimized_labels, remove_cluster_id):


    # クラスタなしデータ（特定クラスタを除外し，最近傍のnodeのclassで置換する）
    unique_cluster = np.unique(optimized_labels)
    feture_graph = create_knn_graph_in_feature_space(embedding, k=5)
    labels_without_cluster = optimized_labels.copy()
    for node in range(len(optimized_labels)):
        if labels_without_cluster[node] == remove_cluster_id:
            ###remove_cluster_id以外からsampling
            candidate_cluster = np.unique(labels_without_cluster)
            candidate_cluster = candidate_cluster[candidate_cluster != remove_cluster_id]
            new_cluster_id = random.sample(list(candidate_cluster),1)[0]
            ###最近傍のnodeのclusterに変更
            for neighbor_node in feture_graph.neighbors(node):
                if optimized_labels[neighbor_node]!=remove_cluster_id:
                    new_cluster_id = optimized_labels[neighbor_node]
                    break
            labels_without_cluster[node] = new_cluster_id
        
    # BIC計算
    bic_with = compute_bic(embedding, optimized_labels)
    bic_without = compute_bic(embedding, labels_without_cluster)

    # 結果表示
    print(f"BIC with cluster {remove_cluster_id}: {bic_with}  BIC without cluster {remove_cluster_id}: {bic_without}")

    return bic_with, bic_without, labels_without_cluster


# クラスタありとなしのデータでBICを計算(再配置するclusterは最近傍の重心を持つcluster
def compare_bic_grav(embedding, optimized_labels, remove_cluster_id):

    n_clusters = len(np.unique(optimized_labels))
    unique_labels = np.unique(optimized_labels)
    labels_without_cluster = optimized_labels.copy()
    
    cluster_centers = np.array([
        embedding[optimized_labels == cluster_id].mean(axis=0) for cluster_id in unique_labels
    ])  # 各クラスタの重心を計算

    # 除外対象のクラスタのデータ点を抽出
    mask_remove = (optimized_labels == remove_cluster_id)
    mask_keep = ~mask_remove
    

    # 除外対象データ点の座標
    remove_points = embedding[mask_remove]

    for node in range(len(labels_without_cluster)):
        temp_cluster =labels_without_cluster[node]
        if temp_cluster == remove_cluster_id:
            new_label = -1
            new_dist = float('inf')
            for cand_cluster in np.unique(optimized_labels):
                if cand_cluster != remove_cluster_id:
                    temp_distance  = np.linalg.norm(cluster_centers[cand_cluster,:] - embedding[node,:])
                    if temp_distance<new_dist:
                        new_label=cand_cluster
                        new_dist=temp_distance
            labels_without_cluster[node]=new_label
            
        


    # BIC計算
    bic_with = compute_bic(embedding, optimized_labels)
    bic_without = compute_bic(embedding, labels_without_cluster)

    # 結果表示
    print(f"BIC with cluster {remove_cluster_id}: {bic_with}  BIC without cluster {remove_cluster_id}: {bic_without}")

    return bic_with, bic_without, labels_without_cluster



# クラスタありとなしのデータでBICを計算(再配置するclusterは最近傍の重心を持つcluster
def compare_silhouette_grav(embedding, optimized_labels, remove_cluster_id):

    n_clusters = len(np.unique(optimized_labels))
    unique_labels = np.unique(optimized_labels)
    labels_without_cluster = optimized_labels.copy()
    
    cluster_centers = np.array([
        embedding[optimized_labels == cluster_id].mean(axis=0) for cluster_id in unique_labels
    ])  # 各クラスタの重心を計算

    # 除外対象のクラスタのデータ点を抽出
    mask_remove = (optimized_labels == remove_cluster_id)
    mask_keep = ~mask_remove
    

    # 除外対象データ点の座標
    remove_points = embedding[mask_remove]

    for node in range(len(labels_without_cluster)):
        temp_cluster =labels_without_cluster[node]
        if temp_cluster == remove_cluster_id:
            new_label = -1
            new_dist = float('inf')
            for cand_cluster in np.unique(optimized_labels):
                if cand_cluster != remove_cluster_id:
                    temp_distance  = np.linalg.norm(cluster_centers[cand_cluster,:] - embedding[node,:])
                    if temp_distance<new_dist:
                        new_label=cand_cluster
                        new_dist=temp_distance
            labels_without_cluster[node]=new_label
            
        


    # silhoette計算
    silhouette_with = silhouette_score(embedding, optimized_labels)
    silhouette_without = silhouette_score(embedding, labels_without_cluster)

    # 結果表示
    print(f"silhouette with cluster {remove_cluster_id}: {silhouette_with}  silhouette without cluster {remove_cluster_id}: {silhouette_without}")

    return silhouette_with, silhouette_without, labels_without_cluster


# クラスタありとなしのデータでBICを計算
def compare_silhouette(embedding, optimized_labels, remove_cluster_id):


    # クラスタなしデータ（特定クラスタを除外し，最近傍のnodeのclassで置換する）
    unique_cluster = np.unique(optimized_labels)
    feture_graph = create_knn_graph_in_feature_space(embedding, k=5)
    labels_without_cluster = optimized_labels.copy()
    for node in range(len(optimized_labels)):
        if labels_without_cluster[node] == remove_cluster_id:
            ###remove_cluster_id以外からsampling
            candidate_cluster = np.unique(labels_without_cluster)
            candidate_cluster = candidate_cluster[candidate_cluster != remove_cluster_id]
            new_cluster_id = random.sample(list(candidate_cluster),1)[0]
            ###最近傍のnodeのclusterに変更
            for neighbor_node in feture_graph.neighbors(node):
                if optimized_labels[neighbor_node]!=remove_cluster_id:
                    new_cluster_id = optimized_labels[neighbor_node]
                    break
            labels_without_cluster[node] = new_cluster_id
        
    # silhoette計算
    silhouette_with = silhouette_score(embedding, optimized_labels)
    silhouette_without = silhouette_score(embedding, labels_without_cluster)

    # 結果表示
    print(f"silhouette with cluster {remove_cluster_id}: {silhouette_with}  silhouette without cluster {remove_cluster_id}: {silhouette_without}")

    return silhouette_with, silhouette_without, labels_without_cluster



def recluster_with_kmeans(embedding, new_optimized_labels, seed = 314):

    
    # クラスタ数はユニークなクラスタ数
    num_clusters = len(np.unique(new_optimized_labels))
    
    # 初期クラスタ中心の計算
    initial_centers = np.array([
        embedding[new_optimized_labels == cluster].mean(axis=0)
        for cluster in np.unique(new_optimized_labels)
    ])
    
    print(f"再クラスタリング開始: クラスタ数 = {num_clusters}")
    
    # KMeansで再クラスタリング
    kmeans = KMeans(n_clusters=num_clusters, init=initial_centers, n_init=1, random_state=seed)
    final_labels = kmeans.fit_predict(embedding)

    print("再クラスタリング完了")

    return final_labels, kmeans




