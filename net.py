import torch.nn as nn
import torch.nn.functional as F
from utils import n_classes, device


class Net(nn.Module):
    def __init__(self, *, device=device):
        super(Net, self).__init__()
        self.to(device=device)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 5, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.4),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.lin = nn.Sequential(
            nn.Linear(128 * 23 ** 2, 128),
            nn.ReLU(128),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 128 * 23 ** 2)
        x = self.lin(x)
        return x
