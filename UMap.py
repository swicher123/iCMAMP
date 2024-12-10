#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/2 17:13
# @Author  : zdj
# @FileName: UMap.py
# @Software: PyCharm

# 实现多标签分类的umap可视化
import numpy as np
import umap.umap_ as umap
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 创建一个多标签分类数据集
X, y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)

# 使用StandardScaler进行数据预处理--标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用UMAP进行降维
reducer = umap.UMAP(n_components=2,random_state=42)
embedding = reducer.fit_transform(X_scaled)

# 使用matplotlib进行可视化
plt.figure(figsize=(10, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the multi-label dataset', fontsize=24)
plt.show()