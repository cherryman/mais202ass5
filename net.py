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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    def __init__(self, dout, *, device=device):
        super(Net, self).__init__()
        self.to(device=device)

        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(1, 256, 3)
        self.conv2 = nn.Conv2d(256, 128, 3)
        self.conv3 = nn.Conv2d(128, 64, 3)
        self.fc1 = nn.Linear(64 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, dout)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x