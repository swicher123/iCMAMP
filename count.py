#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/29 14:23
# @Author  : zdj
# @FileName: count.py
# @Software: PyCharm
with open('data/train_chemical_sequence.txt') as f:
    max_len=0
    for line in f:
        count=0
        for i in line:
            count+=1
        if count>max_len:
            max_len=count
    print(max_len)

