#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/19 11:02
# @Author  : zdj
# @FileName: tSNE.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt_sne
from sklearn import datasets
from sklearn.manifold import TSNE
import os


def plot_tsne(features, labels, epoch, fileNameDir=None):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    # 创建目标文件夹
    if not os.path.exists(fileNameDir):
        os.makedirs(fileNameDir)
    import pandas as pd
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    import seaborn as sns

    # 查看标签的种类有几个
    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4

    try:
        tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    except:
        tsne_features = tsne.fit_transform(features)

    x_min, x_max = np.min(tsne_features, 0), np.max(tsne_features, 0)
    tsne_features = (tsne_features - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    # 一个类似于表格的数据结构
    df = pd.DataFrame()
    df["y"] = labels
    df["comp1"] = tsne_features[:, 0]
    df["comp2"] = tsne_features[:, 1]

    # 颜色是根据标签的大小顺序进行赋色.
    hex = ["#c957db", "#dd5f57", "#b9db57", "#57db30", "#5784db","F1FFED"]  # 绿、红
    data_label = []
    for v in df.y.tolist():
        if v == 0:
            data_label.append("c0")
        elif v == 1:
            data_label.append("c1")
        elif v == 2:
            data_label.append("c2")
        elif v == 3:
            data_label.append("c3")

    df["value"] = data_label

    # hue=df.y.tolist()
    # hue:根据y列上的数据种类，来生成不同的颜色；
    # style:根据y列上的数据种类，来生成不同的形状点；
    # s:指定显示形状的大小
    sns.scatterplot(x=df.comp1.tolist(), y=df.comp2.tolist(), hue=df.value.tolist(), style=df.value.tolist(),
                    palette=sns.color_palette(hex, class_num),
                    markers={"c0": ".", "c1": ".", "c2": ".", "c3": "."},
                    # s = 10,
                    data=df).set(title="")  # T-SNE projection

    # 指定图注的位置 "lower right"
    plt_sne.legend(loc="lower right")
    # 不要坐标轴
    plt_sne.axis("off")
    # 保存图像
    plt_sne.savefig(os.path.join(fileNameDir, "%s.jpg") % str(epoch), format="jpg", dpi=300)
    plt_sne.show()


if __name__ == '__main__':
    digits = datasets.load_digits(n_class=4)
    features, labels = digits.data, digits.target

    plot_tsne(features, labels, "five_visual_2d", fileNameDir="test")