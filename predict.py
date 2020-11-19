import numpy as np
import torch
from net import Net
from utils import norm, show

weights_path = "model-state-dict-epoch-1-31.06-val-loss.pt"

X_test = np.load("test_x.npy")
for im in X_test:
    im[im<220] = 0

X_test = torch.from_numpy(X_test).unsqueeze(1).unsqueeze(1)

y_pred_test = []

net = Net()
net.load_state_dict(torch.load(weights_path))
for i, im in enumerate(X_test):
    print(i)
    y_pred_test.append([int(i), int(torch.argmax(net(im)))])

print(y_pred_test)
# Add header ID, label to csv file
np.savetxt(
    "predictionsbest.csv", y_pred_test, delimiter=",", fmt="%d"
)
