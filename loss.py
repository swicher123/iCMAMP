#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 9:37
# @Author  : zdj
# @FileName: loss.py
# @Software: PyCharm
from keras import backend as K
import tensorflow as tf


def multi_label_asymmtric_loss_tf(labels, logits, gamma_pos=1, gamma_neg=4, clip=0.05, eps=1e-8):
    # 计算概率 caalculating probabilities
    logits_sigmoid = tf.nn.sigmoid(logits)
    logits_sigmoid_pos = logits_sigmoid
    logits_sigmoid_neg = 1 - logits_sigmoid_pos

    # asymmetric clipping
    if clip is not None and clip > 0:
        # logits_sigmoid_neg + clip 有可能大于1
        logits_sigmoid_neg = tf.clip_by_value((logits_sigmoid_neg + clip), clip_value_min=0, clip_value_max=1.0)

    # basic cross entropy
    # logits_sigmoid_pos的取值范围是0-1，因此可以直接可以取对数log，不会溢出
    loss_pos = labels * tf.math.log(tf.clip_by_value(logits_sigmoid_pos, clip_value_min=eps))
    loss_neg = (1 - labels) * tf.math.log(tf.clip_by_value(logits_sigmoid_neg, clip_value_min=eps))
    loss = loss_pos + loss_neg

    # Asymmetric focusing

    if gamma_neg > 0 or gamma_pos > 0:
        pt0 = logits_sigmoid_pos * labels
        pt1 = logits_sigmoid_neg * (1 - labels)
        pt = pt0 + pt1

        one_sided_gamma = gamma_pos * labels + gamma_neg * (1 - labels)
        one_sided_w = tf.pow(1 - pt, one_sided_gamma)
        one_sided_w_no_gradient = tf.stop_gradient([pt0, pt1, pt, one_sided_gamma, one_sided_w])
        loss *= one_sided_w_no_gradient

    return -tf.reduce_sum(loss)


def single_label_asymmetric_loss(labels, logits, gamma_pos=4.0, gamma_neg=0.0, eps: float = 0.1, reduction="mean"):

    num_classes = logits.get_shape().as_list()[-1]

    log_probs = tf.nn.log_softmax(logits)
    # shape = labels.get_shape().as_list()
    origin_target_classes = tf.one_hot(labels, depth=num_classes)

    # asymmetric loss weights
    target_classes = origin_target_classes
    anti_targets = 1 - target_classes

    logits_pos = tf.exp(log_probs)

    logits_neg = 1 - logits_pos
    print("logits_pos: ", logits_pos)
    print("target_classes: ", target_classes)

    logits_pos = tf.multiply(logits_pos, target_classes)
    logits_neg = tf.multiply(logits_neg, anti_targets)

    print("logits_pos: ", logits_pos)

    # logits_pos *= target_classes
    # logits_neg *= anti_targets

    asymmetric_w = tf.pow(1 - logits_pos - logits_neg, gamma_pos * target_classes + gamma_neg * anti_targets)

    log_probs = log_probs * asymmetric_w

    if eps > 0:  # label smoothing
        origin_target_classes = origin_target_classes * (1 - eps) + eps / num_classes

    # loss calculation

    loss = -tf.reduce_sum(tf.multiply(origin_target_classes, log_probs), axis=-1)

    if reduction == "mean":
        loss = tf.reduce_mean(loss)

    return loss
def AsymmetricLoss(y_true,y_pred,alpha=0.5,beta=0.5):

    # 初始化阈值
    # threshold = K.variable(0.5)
    y_pred=tf.nn.sigmoid(y_pred)
    pos_loss = -alpha * tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1.0))  # 正样本损失
    neg_loss = -beta * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-8, 1.0))  # 负样本损失

    # 根据真实标签选择损失
    loss = tf.where(tf.equal(y_true, 1), pos_loss, neg_loss)
    loss=tf.reduce_mean(loss)

    # 添加一个正则项来更新阈值
    # loss+=K.mean(K.abs(threshold-0.5))*0.01

    return loss

def AsymmetricLossOptimizer(y_true,y_pred,alpha=0.7,beta=1):
    y_pred=tf.nn.sigmoid(y_pred)
    # a_y_pred=max((y_pred-0.5),0)
    a_y_pred=tf.where(y_pred>0.5,y_pred-0.5,0)
    # a_y_pred=(y_pred-0.5) if y_pred>0.5 else 0
    pos_loss = - pow((1-y_pred),alpha) * tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1.0))  # 正样本损失
    neg_loss =- pow(a_y_pred,beta)* tf.math.log(tf.clip_by_value(1 - a_y_pred, 1e-8, 1.0))  # 负样本损失
    # pos_loss=pow((1-y_pred),alpha)*tf.math.log(y_pred)
    # neg_loss=pow(a_y_pred,beta)*tf.math.log(1-a_y_pred)

    # 根据真实标签选择损失
    loss = tf.where(tf.equal(y_true, 1), pos_loss, neg_loss)
    loss=tf.reduce_mean(loss)

    return loss