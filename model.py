#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
X_train = np.load("train_x.npy")
y_train = pd.read_csv("train_y.csv", index_col='ID').to_numpy()
X_test = np.load("test_x.npy")

def show(arr):
    two_d = (np.reshape(arr, (128, 128)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    plt.show()

show(X_train[0])

model_conv = torchvision.models.vgg19(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False         # freeze weights for pretrained layers, maybe we only want to freeze first few layers and allow rest to be trained?

model_conv.classifier = nn.Sequential(  # replace classifier layers so can retrain them to classify our images
    nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10)         # 10 is number of classes we have
    )
