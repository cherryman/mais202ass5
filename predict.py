#!/usr/bin/env python

import numpy as np
import torch
from net import Net
from utils import device, show

weights_path = "deeper-network-epoch-18-718.74-val-loss-0.9430666666666667-val-accuracy.pt"
net = Net()
net.load_state_dict(torch.load(weights_path))
net.to(device)
net.eval()

X_test = np.load("test_x.npy")
for im in X_test:
    im[im < 220] = 0

show(X_test[0])
X_test = torch.from_numpy(X_test).unsqueeze(1).unsqueeze(1).to(device)
y_preds = []

with torch.no_grad():
    for i, im in enumerate(X_test):
        y_preds.append([int(i), int(torch.argmax(net(im)))])

# Add header "ID,label" to csv file
np.savetxt("predictions7.csv", y_preds, delimiter=",", fmt="%d")
