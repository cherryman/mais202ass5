#!/usr/bin/env python
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from utils import Dataset, norm, device, show, weights_init
from net import Net

FILEPATH = "deeper-network"

# Not calling .to(device) to not run out of memory
# norm calls .from_numpy
X_train = np.load("15000balanced_sorted_x_train.npy")
y_train = np.load("15000balanced_sorted_y_train.npy")

X_train = torch.from_numpy(X_train).unsqueeze(1)
y_train = torch.from_numpy(y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=22, shuffle=True)

net = Net().to(device)

train_loader = utils.data.DataLoader(Dataset(X_train, y_train), batch_size=32)
val_loader = utils.data.DataLoader(Dataset(X_val, y_val), batch_size=16)

learning_rate = 0.001
previous_accuracy = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9,weight_decay=0.01)
net.apply(weights_init)
epochs = 200
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")

    net.train()
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

    # Turn off gradient computations for validation set
    # and put the net into evaluation mode.
    net.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        incorrect = 0
        for i, (x, y) in enumerate(val_loader, 1):
            x, y = x.to(device), y.to(device)
            y_preds = net(x)
            loss = criterion(y_preds, y)
            running_loss += loss.item()
            for index, item in enumerate(y_preds):
                if torch.argmax(item) == y[index]:
                    correct += 1
                else:
                    incorrect += 1
    current_accuracy = correct / (incorrect + correct)
    if previous_accuracy > current_accuracy:
        learning_rate = learning_rate / 2
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.01)
    else:
        previous_accuracy = current_accuracy

    print("Validation loss: {:5.2f}".format(running_loss))
    print("Validation accuracy: {:5.4f}".format(current_accuracy))
    torch.save(
        net.state_dict(),
        f"{FILEPATH}-epoch-{epoch + 1}-{running_loss:5.2f}-val-loss-{current_accuracy}-val-accuracy.pt",
    )
