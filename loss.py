#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 9:37
# @Author  : zdj
# @FileName: loss.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.keras import backend as K

# Focal Loss
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        # Ensure that y_true is of float32 type
        y_true = K.cast(y_true, dtype=tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())  # Avoid log(0) calculation issue

        # Calculate cross-entropy loss
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)

        # Calculate weight
        weight = alpha * K.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * K.pow(y_pred, gamma) * (1 - y_true)

        # Calculate Focal Loss
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=1))  # Sum over samples and take the mean

    return focal_loss_fixed
