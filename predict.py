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

weights_path = "nodropoutweights/model-state-dict-epoch-6-108.80-val-loss.pt"

n_classes = 10
X_test = np.load("test_x.npy")
X_test = torch.from_numpy(X_test)

X_test = X_test.unsqueeze(1)
X_test = X_test.unsqueeze(1)
print(X_test.shape)
y_pred_test = []

net = Net(n_classes)
net.load_state_dict(torch.load(weights_path))
i = 0
for im in X_test[:100]:
    print(i)
    y_pred_test.append([int(i),int(torch.argmax(net(im)))])
    i+=1

print(y_pred_test)
np.savetxt("predictions.csv", y_pred_test, delimiter=",",fmt="%d") # ADD HEADER ID, label to csv file
    