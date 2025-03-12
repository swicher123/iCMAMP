#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/13 10:15
# @Author  : zdj
# @FileName: model.py
# @Software: PyCharm

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten
from keras.layers import Embedding, Dropout
from tensorflow.keras.models import Model



def CNN(input_shape, num_labels):
    main_input = Input(shape=(input_shape,), dtype='float32', name='main_input')

    x = main_input[..., None]

    # 1D 卷积可以保留，但要调整参数
    x = Conv1D(64, kernel_size=2, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)  # Dropout 位置调整
    output = Dense(num_labels, activation='sigmoid', name='output')(x)

    model = Model(inputs=main_input, outputs=output)
    return model
