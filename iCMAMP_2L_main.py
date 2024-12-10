#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/10 10:06
# @Author  : zdj
# @FileName: main.py
# @Software: PyCharm
import csv
import os
import time
from pathlib import Path
import numpy as np
# from train import train_main
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
modelDir = 'model'
Path(modelDir).mkdir(exist_ok=True)
t = time.localtime(time.time())

def staticTrainandTest(y_train, y_test):
    # static number
    data_size_tr = np.zeros(5)
    data_size_te = np.zeros(5)

    for i in range(len(y_train)):
        for j in range(len(y_train[i])):
            if y_train[i][j] > 0:
                data_size_tr[j] += 1

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] > 0:
                data_size_te[j] += 1

    for i in range(5):
        print('{}\n'.format(int(data_size_tr[i])))

    print("TestingSet:\n")
    for i in range(5):
        print('{}\n'.format(int(data_size_te[i])))

    return data_size_tr


def getSequenceData(first_dir, file_name):
    data_path = "{}/{}".format(first_dir, file_name)
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过第一行（即header）
        feature = list(reader)
    return feature


def getFeature(first_dir, file_name):
    path = os.path.join(first_dir, file_name)
    df = pd.read_csv(path)
    subset = df.iloc[:, 0:20]
    return subset


def getLabelData(first_dir, file_name):
    label_list = []
    label_path = "{}/{}.txt".format(first_dir, file_name)
    with open(label_path) as f:
        for each in f:
            each = each.strip()
            label_list.append(np.array(list(each), dtype=int))
    return label_list


def TrainAndTest(tr_data, tr_label, te_data, te_label, data_size):
    # Call training method
    train = [tr_data, tr_label]
    test = [te_data, te_label]
    threshold = 0.5
    model_num = 10  # model number
    test.append(threshold)

    # train_main(train, test, model_num, modelDir, data_size)

    tt = time.localtime(time.time())
    with open(os.path.join(modelDir, 'time.txt'), 'a+') as f:
        f.write('finish time: {}m {}d {}h {}m {}s'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))


def main():
    first_dir = 'data'
    feature_dir = 'feature_data'

    # train_sequence_data=getFeature(feature_dir,'Mutilabel_feature_train.csv')
    train_sequence_data = getSequenceData(feature_dir, 'DPC_train.csv')
    train_sequence_label = getLabelData(first_dir, 'train_label')
    # test_sequence_data=getFeature(feature_dir,'Mutilabel_feature_test.csv')
    test_sequence_data = getSequenceData(feature_dir, 'DPC_test.csv')
    test_sequence_label = getLabelData(first_dir, 'test_label')

    # Converting the list collection to an array
    x_train = np.array(train_sequence_data)
    x_test = np.array(test_sequence_data)
    y_train = np.array(train_sequence_label)
    y_test = np.array(test_sequence_label)

    # Counting the number of each peptide in the training set and the test set, and return the total number of the training set
    data_size = staticTrainandTest(y_train, y_test)

    # training and predicting the data
    TrainAndTest(x_train, y_train, x_test, y_test, data_size)


if __name__ == '__main__':
    main()
