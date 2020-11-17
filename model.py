#!/usr/bin/env python
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import torchvision
from net import Net
FILEPATH = "model-state-dict"
'''Params to try changing:
-   threshold for determining if white on black or black on white
-   threshold for cutting off pixel noise
-   Learning rate
-   momentum

'''
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

n_classes = 10

y_train = pd.read_csv("train_y.csv", index_col="ID").to_numpy()
y_train = np.reshape(y_train,(len(y_train),)) # turn y_train into one-dimensional vector

X_train = np.load("train_x.npy")

for im in X_train:  # remove noise from images, probably needs some work
    im= im/im.max()
    mean = np.mean(im)
    if mean < 0.7:
        im[im < 0.8] = 0
    else:
        im = 1 - im
        im[im < 0.8] = 0

# X_train = np.repeat(X_train[:,np.newaxis,:,:],3, 1) # Triple channels to work with dimensions of pretrained weights

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05)


# Not calling .to(device) to not run out of memory
X_train = torch.from_numpy(X_train)
X_val = torch.from_numpy(X_val)
y_train = torch.from_numpy(y_train)

X_train = X_train.unsqueeze(1)  # add artificial 4th dimension to fit dimensions needed for conv layer
X_val = X_val.unsqueeze(1)
# model_conv = torchvision.models.vgg19(pretrained=True)
# for param in model_conv.parameters():
#     # freeze weights for pretrained layers, maybe we only want to freeze first few
#     # layers and allow rest to be trained?
#     param.requires_grad = False

# # replace classifier layers so can retrain them to classify our images
# model_conv.classifier = nn.Sequential(
#     nn.Linear(512 * 7 * 7, 1024),
#     nn.ReLU(True),
#     nn.Dropout(),
#     nn.Linear(1024, 512),
#     nn.ReLU(True),
#     nn.Dropout(),
#     nn.Linear(512, n_classes),
# )

# model_conv.to(device)

net = Net(n_classes).to(device)



class Dataset(utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)


train_loader = utils.data.DataLoader(Dataset(X_train, y_train), batch_size=16)
val_loader = utils.data.DataLoader(Dataset(X_val, y_val), batch_size=16)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001,momentum=0.9)

epochs = 20
for epoch in range(epochs):
    print("Epoch {}".format(epoch + 1))
    running_loss = 0.0
    running_losses = []
    for i, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(device), y.to(device)


        optimizer.zero_grad()
        y_preds = net(x)
        loss = criterion(y_preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 0:
            running_losses.append(running_loss)
            average_loss = np.mean(running_losses)
            print(
                "[{:2}, {:5}, {:3.0f}%] loss: {:5.2f}".format(
                    epoch + 1,
                    i,
                    100.0 * (i / len(train_loader) + epoch) / epochs,
                    average_loss,
                )
            )
            
            running_loss = 0.0
    with torch.no_grad():                   # Turn off gradient computations for validation set
        running_loss = 0.0
        correct = 0
        incorrect = 0
        for i, (x, y) in enumerate(val_loader, 1):
            x, y = x.to(device), y.to(device)
            y_preds = net(x)
            loss = criterion(y_preds, y)
            running_loss += loss.item()
            for index, item in enumerate(y_preds):
                if (torch.argmax(item) == y[index]):
                    correct += 1
                else:
                    incorrect += 1
                

    print("Validation loss: {:5.2f}".format(running_loss))      # maybe need to scale validation loss to be proportional to train loss but not really necessary
    print("Validation accuracy: {:5.4f}".format(correct/(incorrect + correct))) 
    torch.save(net.state_dict(), "{}-epoch-{}-{:5.2f}-val-loss.pt".format(FILEPATH, epoch + 1, running_loss))
