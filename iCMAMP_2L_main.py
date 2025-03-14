#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/10 10:06
# @Author  : zdj
# @FileName: main.py
# @Software: PyCharm
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from model import CNN
from evaluation import evaluate
from tensorflow.keras.optimizers import Adam
from loss import focal_loss

# Set random seed for reproducibility
def set_random_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_random_seed()


# Function to read features
def read_csv_feature(first_dir, file_name, usecols=None):
    path = os.path.join(first_dir, file_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    try:
        df = pd.read_csv(path) 
        return df.iloc[:, usecols] if usecols else df
    except Exception as e:
        raise RuntimeError(f"Error reading the CSV file: {e}")

# Function to read labels
def get_labels(first_dir, file_name):
    labels = []
    label_path = os.path.join(first_dir, f"{file_name}.txt")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label file {label_path} does not exist.")
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            labels.append(np.array([int(x) for x in line.split(',')]))
    return np.array(labels)

# Convert string type features to float
def convert_str_to_float(*arrays):
    converted = [arr.astype(float) if np.issubdtype(arr.dtype, np.str_) else arr for arr in arrays]
    return converted[0] if len(converted) == 1 else converted  # Return the array if only one array is passed

# Read data
first_dir = 'data'
threshold = 0.5
train_feature = read_csv_feature('feature_data', 'DPC_train.csv').values
test_feature = read_csv_feature('feature_data', 'DPC_test.csv').values
y_train = get_labels(first_dir, 'train_label')
y_test = get_labels(first_dir, 'test_label')


X_train = train_feature
# Ensure data types are correct
X_train, y_train = convert_str_to_float(X_train, y_train)
X_test = test_feature
X_test, y_test = convert_str_to_float(X_test, y_test)

# Normalize features
input_dim = X_train.shape[1]
num_labels = y_train.shape[1]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
precision_val, coverage_val, accuracy_val, absolute_true_val, absolute_false_val = [], [], [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Training fold {fold + 1}...")
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    # Initialize model
    model = CNN(input_shape=input_dim, num_labels=num_labels)
    optimizer = Adam(learning_rate=0.001)  # Set learning rate
    model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])

    # Train model
    model.fit(X_train_fold, y_train_fold, batch_size=32, epochs=100, validation_data=(X_val_fold, y_val_fold))

    # Evaluate model
    y_val_pred = model.predict(X_val_fold) > threshold
    precision_train, coverage_train, accuracy_train, absolute_true_train, absolute_false_train = evaluate(
        y_val_pred.astype(int), y_val_fold)

    # Append results
    precision_val.append(precision_train)
    coverage_val.append(coverage_train)
    accuracy_val.append(accuracy_train)
    absolute_true_val.append(absolute_true_train)
    absolute_false_val.append(absolute_false_train)

# Train set results
train_results = f"""
===== Train Set Results =====
Precision: {np.mean(precision_val)}
Coverage: {np.mean(coverage_val)}
Accuracy: {np.mean(accuracy_val)}
Absolute True: {np.mean(absolute_true_val)}
Absolute False: {np.mean(absolute_false_val)}
"""

# Model training
final_model = CNN(input_shape=input_dim, num_labels=num_labels)
optimizer = Adam(learning_rate=0.001)  # Set learning rate
final_model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])
final_model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))
y_pred = final_model.predict(X_test) > threshold
precision_test, coverage_test, accuracy_test, absolute_true_test, absolute_false_test = evaluate(y_pred.astype(int), y_test)

# Test set results
test_results = f"""
===== Test Set Results =====
Precision: {precision_test}
Coverage: {coverage_test}
Accuracy: {accuracy_test}
Absolute True: {absolute_true_test}
Absolute False: {absolute_false_test}
"""

print(train_results)
print(test_results)

# Save evaluation results to file
with open("result.txt", "a+") as f:
    f.write(train_results)
    f.write(test_results)

print("Evaluation results have been saved to result.txt")
