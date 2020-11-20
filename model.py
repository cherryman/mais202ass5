#!/usr/bin/env python
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from utils import Dataset, norm, device
from net import Net

FILEPATH = "model-state-dict"

# Not calling .to(device) to not run out of memory
# norm calls .from_numpy
X_train = norm(np.load("train_x.npy"))
y_train = pd.read_csv("train_y.csv", index_col="ID").to_numpy().reshape((-1))
y_train = torch.from_numpy(y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05)

net = Net().to(device)

train_loader = utils.data.DataLoader(Dataset(X_train, y_train), batch_size=32)
val_loader = utils.data.DataLoader(Dataset(X_val, y_val), batch_size=16)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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

    print("Validation loss: {:5.2f}".format(running_loss))
    print("Validation accuracy: {:5.4f}".format(correct / (incorrect + correct)))
    torch.save(
        net.state_dict(),
        f"{FILEPATH}-epoch-{epoch + 1}-{running_loss:5.2f}-val-loss.pt",
    )
