import numpy as np
import torch
from net import Net
from utils import norm

weights_path = "nodropoutweights/model-state-dict-epoch-6-108.80-val-loss.pt"

X_test = norm(np.load("test_x.npy"))
y_pred_test = []

net = Net()
net.load_state_dict(torch.load(weights_path))
for i, im in enumerate(X_test):
    print(i)
    y_pred_test.append([int(i), int(torch.argmax(net(im)))])

print(y_pred_test)
# Add header ID, label to csv file
np.savetxt(
    "predictions.csv", y_pred_test, delimiter=",", fmt="%d"
)
