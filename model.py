#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/13 10:15
# @Author  : zdj
# @FileName: model.py
# @Software: PyCharm

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten
from keras.layers import Embedding, Dropout
from tensorflow.keras.models import Model



def CNN(input_shape, num_labels, length):
    main_input = Input(shape=(input_shape,), dtype='int64', name='main_input')
    x = Embedding(output_dim=128, input_dim=200, input_length=length, name='Embadding')(main_input)

    a = Conv1D(64, 2, activation='relu', padding='same')(x)
    apool = MaxPooling1D(pool_size=2, strides=1, padding='same')(a)
    merge = Dropout(0.6)(apool)

    x = Flatten()(merge)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_labels, activation='sigmoid', name='output')(x)
    model = Model(inputs=main_input, outputs=output)

    return model
