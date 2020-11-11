#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train = np.load("train_x.npy")
y_train = pd.read_csv("train_y.csv", index_col='ID').to_numpy()
X_test = np.load("test_x.npy")

def show(arr):
    two_d = (np.reshape(arr, (128, 128)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    plt.show()

show(X_train[0])
