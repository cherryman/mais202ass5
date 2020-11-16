#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torchvision

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_classes = 10

X_train = np.load("train_x.npy")
X_train = np.repeat(X_train[:,np.newaxis,:,:],3, 1) # Triple channels to work with dimensions of pretrained weights
X_train = X_train/X_train.max() # Normalize input, maybe should do with stdev instead?

y_train = pd.read_csv("train_y.csv", index_col="ID").to_numpy()
y_train = np.reshape(y_train,(len(y_train),)) # turn y_train into one-dimensional vector

X_test = np.load("test_x.npy")



# Not calling .to(device) to not run out of memory
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)

model_conv = torchvision.models.vgg19(pretrained=True)
for param in model_conv.parameters():
    # freeze weights for pretrained layers, maybe we only want to freeze first few
    # layers and allow rest to be trained?
    param.requires_grad = False

# replace classifier layers so can retrain them to classify our images
model_conv.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, n_classes),
)

model_conv.to(device)


def show(arr):
    two_d = (np.reshape(arr, (128, 128)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation="nearest")
    plt.show()


class Dataset(utils.data.Dataset):
    def __getitem__(self, i):
        return X_train[i], y_train[i]

    def __len__(self):
        return len(y_train)


loader = utils.data.DataLoader(Dataset(), batch_size=16)
net = model_conv
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epochs = 20
for epoch in range(epochs):
    print("Epoch {}".format(epoch))
    running_loss = 0.0
    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)


        optimizer.zero_grad()
        y_preds = net(x)
        loss = criterion(y_preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 0:
            print(
                "[{:2}, {:5}, {:3.0f}%] loss: {:5.2f}".format(
                    epoch + 1,
                    i,
                    100.0 * (i / len(loader) + epoch) / epochs,
                    running_loss,
                )
            )
            running_loss = 0.0
