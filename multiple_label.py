#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/5 9:31
# @Author  : zdj
# @FileName: multiple_label.py
# @Software: PyCharm

import csv
import os


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from loss import multi_label_asymmtric_loss_tf, AsymmetricLoss, AsymmetricLossOptimizer

from datetime import datetime
from keras import backend as K
from keras.layers import Layer
from keras import initializers
import keras.optimizers as optimizers

import numpy as np
import pandas as pd
from keras.layers import Embedding, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Attention, concatenate, \
    Flatten
from tensorflow.keras.optimizers import Adam
from evaluation import evaluate
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# 读取第一个特征
def getFeature1(first_dir, file_name):
    path = os.path.join(first_dir, file_name)
    df = pd.read_csv(path)
    subset = df.iloc[:, 0:20]
    # subset1=df.iloc[:,0:20]
    # new_subset=pd.concat([subset1,subset],axis=1)
    return subset
    # return new_subset

# 读取第二个特征
def getFeature2(first_dir, file_name):
    path = os.path.join(first_dir, file_name)
    df = pd.read_csv(path)
    subset = df.iloc[:, :]
    return subset

# 读取第三个特征
def getFeature3(first_dir, file_name):
    path = os.path.join(first_dir, file_name)
    df = pd.read_csv(path)
    subset = df.iloc[:, 70:470]
    return subset


def getLabelData(first_dir, file_name):
    label_list = []
    label_path = "{}/{}.txt".format(first_dir, file_name)
    with open(label_path) as f:
        for each in f:
            each = each.strip()
            label_list.append(np.array(list(each), dtype=int))
    return label_list


def convert_str_to_int(arr1, arr2):
    has_str = False
    for arr in [arr1, arr2]:
        if np.issubdtype(arr.dtype, np.str_):
            has_str = True
            break

    if has_str:
        arr1 = arr1.astype(float, copy=False)
        arr2 = arr2.astype(float, copy=False)

    return arr1, arr2


train_data_path = 'feature_data/Mutilabel_feature_train.csv'
test_data_path = 'feature_data/Mutilabel_feature_test.csv'
first_dir = 'data'
threshold = 0.5

"""
with open(train_data_path,'r',encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过第一行（即header）
    X_train =list(reader)
with open(test_data_path,'r',encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过第一行（即header）
    X_test =list(reader)
"""

# 读取特征1的数据(已经归一化)
train_feature_1 = np.array(getFeature1('feature_data', 'Mutilabel_feature_train.csv'))

# 读取特征2的数据
train_feature_2=getFeature2('feature_data', 'esm1b.csv')
# print(feature_2)
# 对特征2的数据进行归一化
scaler = MinMaxScaler(feature_range=(0,100))
scaler.fit(train_feature_2)
train_feature_2 = scaler.transform(train_feature_2)
# print(feature_2)

# 读取特征3的数据（已经归一化）
train_feature_3 = np.array(getFeature3('feature_data', 'Mutilabel_feature_train.csv'))


# 拼接特征
X_train = np.concatenate((train_feature_1, train_feature_2, train_feature_3), axis=1)
print(X_train)
print(X_train.shape)
# X_train = getFeature('feature_data', 'Mutilabel_feature_train.csv')
# X_test = getFeature('feature_data', 'Mutilabel_feature_test.csv')

X_train = np.array(X_train)
X_test = np.array(X_test)

train_label = getLabelData(first_dir, 'train_label')
test_label = getLabelData(first_dir, 'test_label')

y_train = np.array(train_label)
y_test = np.array(test_label)


class MultiHeadAttention(Layer):
    def __init__(self, output_dim, num_head, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.num_head = num_head
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        return {"output_dim": self.output_dim, "num_head": self.num_head}

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(self.num_head, 3, input_shape[2], self.output_dim),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        self.Wo = self.add_weight(name='Wo',
                                  shape=(self.num_head * self.output_dim, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        self.built = True

    def call(self, x):
        q = K.dot(x, self.W[0, 0])
        k = K.dot(x, self.W[0, 1])
        v = K.dot(x, self.W[0, 2])
        e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘
        e = e / (self.output_dim ** 0.5)
        e = K.softmax(e)
        outputs = K.batch_dot(e, v)
        for i in range(1, self.W.shape[0]):
            q = K.dot(x, self.W[i, 0])
            k = K.dot(x, self.W[i, 1])
            v = K.dot(x, self.W[i, 2])
            # print('q_shape:'+str(q.shape))
            e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘
            e = e / (self.output_dim ** 0.5)
            e = K.softmax(e)
            # print('e_shape:'+str(e.shape))
            o = K.batch_dot(e, v)
            outputs = K.concatenate([outputs, o])
        z = K.dot(outputs, self.Wo)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


length = X_train.shape[1]
out_length = y_train.shape[1]


# 构建模型
def build_model(input_shape, num_labels):
    main_input = Input(shape=(input_shape,), dtype='int64', name='main_input')
    x = Embedding(output_dim=128, input_dim=200, input_length=length, name='Embadding')(main_input)

    a = Conv1D(64, 2, activation='relu', padding='same')(x)
    apool = MaxPooling1D(pool_size=2, strides=1, padding='same')(a)

    merge = Dropout(0.6)(apool)

    # bilstm=Bidirectional(LSTM(64,return_sequences=True))(apool)
    # bilstm = Bidirectional(LSTM(64, return_sequences=True))(merge)
    # x = MultiHeadAttention(80, 5)(bilstm)

    # x = Flatten()(x)
    # x=Flatten()(bilstm)
    # x=Flatten()(apool)
    x = Flatten()(merge)
    x = Dense(128, activation='relu')(x)

    output = Dense(num_labels, activation='sigmoid', name='output')(x)
    # output = Dense(num_labels, name='output')(x)

    model = Model(inputs=main_input, outputs=output)
    # adam = optimizers.adam_v2.Adam(lr=lr)
    # 修改
    # adam = optimizers.adam_v2.Adam(learning_rate=0.001)
    # model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    # model.compile(loss=AsymmetricLossOptimizer,optimizer=Adam(),metrics=['accuracy'])

    # model.summary()

    return model


model = build_model(input_shape=length, num_labels=y_train.shape[1])

X_train, y_train = convert_str_to_int(X_train, y_train)
X_test, y_test = convert_str_to_int(X_test, y_test)

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=1)
precision_val, coverage_val, accuracy_val, absolute_true_val, absolute_false_val = [], [], [], [], []
# k折交叉验证
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    model.fit(X_train_fold, y_train_fold)
    y_val_pred = model.predict(X_val_fold)

    y_val_pred = (y_val_pred > threshold).astype(int)
    """
    train_thresholds = []
    for i in range(y_val_pred.shape[1]):
        pos_pred = y_val_pred[:, i][y_val_fold[:, i] == 1].mean()
        neg_pred = y_val_pred[:, i][y_val_fold[:, i] == 0].mean()
        train_thresholds.append((pos_pred + neg_pred) / 2)
    print(f"各标签阈值平均值：{train_thresholds}")
    y_val_pred = (y_val_pred > np.array(train_thresholds)).astype(int)
    """

    precision_train, coverage_train, accuracy_train, absolute_true_train, absolute_false_train = evaluate(y_val_pred,
                                                                                                          y_val_fold)
    precision_val.append(precision_train)
    coverage_val.append(coverage_train)
    accuracy_val.append(accuracy_train)
    absolute_true_val.append(absolute_true_train)
    absolute_false_val.append(absolute_false_train)

# 训练集的评估
precision_train = sum(precision_val) / k
coverage_train = sum(coverage_val) / k
accuracy_train = sum(accuracy_val) / k
absolute_true_train = sum(absolute_true_val) / k
absolute_false_train = sum(absolute_false_val) / k

print("Training ....")
print("Precisions:", precision_train)
print("Coverage:", coverage_train)
print("Accuracy:", accuracy_train)
print("Absolute True:", absolute_true_train)
print("Absolute False:", absolute_false_train)

index = np.arange(len(y_train))
np.random.shuffle(index)
X_train = X_train[index]
y_train = y_train[index]
# 训练模型
model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_test, y_test))

# 评估模型
y_pred = model.predict(X_test)
y_pred = (y_pred > threshold).astype(int)
print(type(y_pred))
"""
# 修改
threslist = [0.05,0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
pre,cov,acc,abst,absf=[],[],[],[],[]
for thre in threslist:
    # y_pred = model.predict(X_test)
    y_pred=(y_pred>thre).astype(int)
    precision, coverage, accuracy, absolute_true, absolute_false = evaluate(y_pred, y_test)
    pre.append(precision)
    cov.append(coverage)
    acc.append(accuracy)
    abst.append(absolute_true)
    absf.append(absolute_false)
target=argmax(np.array(abst))
print("target:",target)
print("threshold",threslist[target])
print("Testing ....")
print("Precisions:", pre[target])
print("Coverage:", cov[target])
print("Accuracy:", acc[target])
print("Absolute True:", abst[target])
print("Absolute False:", absf[target])
"""

"""
thresholds = []
for i in range(y_pred.shape[1]):
    pos_pred=y_pred[:,i][y_test[:,i]==1].mean()
    neg_pred=y_pred[:,i][y_test[:,i]==0].mean()
    thresholds.append((pos_pred+neg_pred)/2)
print(f"各标签阈值平均值：{thresholds}")
y_pred=(y_pred>np.array(thresholds)).astype(int)
"""
precision, coverage, accuracy, absolute_true, absolute_false = evaluate(y_pred, y_test)

# 打印结果
print("Testing ....")
print("Precisions:", precision)
print("Coverage:", coverage)
print("Accuracy:", accuracy)
print("Absolute True:", absolute_true)
print("Absolute False:", absolute_false)

# n_labels 参数指定了每个样本看可以拥有的标签数量
# n_labels=5
# tsne可视化
tsne=TSNE(n_components=2,perplexity=30,random_state=0)
x_tsne=tsne.fit_transform(X_test)
"""
n_labels=22
colors = plt.cm.get_cmap('viridis', n_labels)
for i, (x, y, labels) in enumerate(zip(x_tsne[:, 0], x_tsne[:, 1], y_pred)):
    # 获取非零标签的索引，即样本拥有的标签
    label_indices = np.where(labels)[0]
    # 如果样本有标签，使用这些标签的颜色的平均值进行着色
    if label_indices.size > 0:
        color = colors(np.mean(label_indices / (n_labels - 1)))
    else:
        color = 'gray'  # 没有标签的样本使用灰色
    plt.scatter(x, y, color=color, alpha=0.8)
# 添加标题
plt.title('t-SNE Visualization with Multi-label Data')

# 显示图形
plt.show()
"""

"""
plt.figure(figsize=(10, 8))
# colors = ['red', 'blue', 'green', 'purple', 'orange']

# 为每个标签绘制散点图
for i in range(n_labels):
    class_member_mask = y_pred[:, i] == 1
    xy = x_tsne[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], alpha=0.8, cmap='viridis', label=f'Label {i + 1}')
# 添加图例
plt.legend()

# 添加标题
plt.title('t-SNE Visualization of Multi-label Classification Dataset')

# 显示图形
plt.show()
"""

new_label_list=[]
for row in y_pred:
    if (row==np.array([0,0,0,0,0])).all():
        new_label=0
        new_label_list.append(new_label)
    elif (row==np.array([0,0,0,1,0])).all():
        new_label=1
        new_label_list.append(new_label)
    elif (row==np.array([0,0,0,0,1])).all():
        new_label=2
        new_label_list.append(new_label)
    elif (row==np.array([1,0,0,0,0])).all():
        new_label=3
        new_label_list.append(new_label)
    elif (row==np.array([0,1,1,0,0])).all():
        new_label=4
        new_label_list.append(new_label)
    elif (row==np.array([0,1,0,0,1])).all():
        new_label=5
        new_label_list.append(new_label)
    elif (row==np.array([0,0,1,0,0])).all():
        new_label=6
        new_label_list.append(new_label)
    elif (row==np.array([0,0,1,0,1])).all():
        new_label=7
        new_label_list.append(new_label)
    elif (row==np.array([0,1,1,0,1])).all():
        new_label=8
        new_label_list.append(new_label)
    elif (row==np.array([0,0,0,1,1])).all():
        new_label=9
        new_label_list.append(new_label)
    elif (row==np.array([1,1,0,0,1])).all():
        new_label=10
        new_label_list.append(new_label)
    elif (row==np.array([0,0,1,1,1])).all():
        new_label=11
        new_label_list.append(new_label)
    elif (row==np.array([1,1,1,1,1])).all():
        new_label=12
        new_label_list.append(new_label)
    elif (row==np.array([0,1,0,1,1])).all():
        new_label=13
        new_label_list.append(new_label)
    elif (row==np.array([0,1,1,1,1])).all():
        new_label=14
        new_label_list.append(new_label)
    elif (row==np.array([0,0,1,1,0])).all():
        new_label=15
        new_label_list.append(new_label)
    elif (row==np.array([1,0,1,1,1])).all():
        new_label=16
        new_label_list.append(new_label)
    elif (row==np.array([1,1,1,0,1])).all():
        new_label=17
        new_label_list.append(new_label)
    elif (row==np.array([1,0,0,0,1])).all():
        new_label=18
        new_label_list.append(new_label)
    elif (row==np.array([1,0,1,0,1])).all():
        new_label=19
        new_label_list.append(new_label)
    elif (row==np.array([0,1,0,0,0])).all():
        new_label=20
        new_label_list.append(new_label)
    elif (row==np.array([1,0,1,0,0])).all():
        new_label=21
        new_label_list.append(new_label)
print(new_label_list)



labels=np.array(new_label_list)

highlighted_labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
name_label=['label_1','label_2','label_3']
for i, (x, y, label) in enumerate(zip(x_tsne[:, 0], x_tsne[:, 1], labels)):
    if label==highlighted_labels[3]:
        color = 'blue'
    elif label==highlighted_labels[2]:
        color = 'orange'
    elif label==highlighted_labels[20]:
        color = 'green'
    # elif label==highlighted_labels[9]:
    #     color = 'red'
    # elif label==highlighted_labels[4]:
    #     color = 'purple'
    # elif label==highlighted_labels[5]:
    #     color = 'brown'
    # elif label==highlighted_labels[6]:
    #     color = 'pink'
    # elif label==highlighted_labels[7]:
    #     color = 'yellow'
    # elif label==highlighted_labels[8]:
    #     color = '#C21DFF'
    # elif label==highlighted_labels[9]:
    #     color = '#FE22FF'
    else:
        color = 'gray'
    plt.scatter(x, y, color=color, alpha=0.8)

plt.legend()
plt.show()



now = datetime.now()
current_time = now.strftime('%Y-%m-%d %H:%M:%S')
with open("result/results.txt", 'a+') as f:
    f.write(current_time)
    # f.writelines(str(thresholds))
    f.close()

data = []
data.append('\nTrainsets:')
data.append('precision:{}'.format(str(precision_train)))
data.append('coverage:{}'.format(str(coverage_train)))
data.append('accuracy:{}'.format(str(accuracy_train)))
data.append('absolute_true:{}'.format(str(absolute_true_train)))
data.append('absolute_false:{}'.format(str(absolute_false_train)))

data.append('Testsets:')
data.append('precision:{}'.format(str(precision)))
data.append('coverage:{}'.format(str(coverage)))
data.append('accuracy:{}'.format(str(accuracy)))
data.append('absolute_true:{}'.format(str(absolute_true)))
data.append('absolute_false:{}'.format(str(absolute_false)))
data.append('\n')

with open("result/results.txt", 'ab') as x:
    np.savetxt(x, np.asarray(data), fmt="%s\t")

"""
# 特征消融实验
X_ablated_train = X_train.copy()
X_ablated_test = X_test.copy()
# 移除特征
X_ablated_train[:,70:470]=0
X_ablated_test[:,70:470]=0
# 初始化并训练消融后的模型
ablated_model = build_model(input_shape=X_ablated_train.shape[1], num_labels=y_train.shape[1])
ablated_model.fit(X_ablated_train,y_train,batch_size=32, epochs=30)
ablated_y_pred=ablated_model.predict(X_ablated_test)
ablated_y_pred = (ablated_y_pred > threshold).astype(int)
a_precision, a_coverage, a_accuracy, a_absolute_true, a_absolute_false = evaluate(ablated_y_pred, y_test)

print("Ablated ....")
print("a_Precisions:", a_precision)
print("a_Coverage:", a_coverage)
print("a_Accuracy:", a_accuracy)
print("a_Absolute True:", a_absolute_true)
print("a_Absolute False:", a_absolute_false)

data1 = []
data1.append('Ablated/3:')
data1.append('a_precision:{}'.format(str(a_precision)))
data1.append('a_coverage:{}'.format(str(a_coverage)))
data1.append('a_accuracy:{}'.format(str(a_accuracy)))
data1.append('a_absolute_true:{}'.format(str(a_absolute_true)))
data1.append('a_absolute_false:{}'.format(str(a_absolute_false)))
data1.append('\n')

with open("result/results.txt", 'ab') as x:
    np.savetxt(x, np.asarray(data1), fmt="%s\t")
"""
