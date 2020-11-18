import matplotlib.pyplot as plt
import torch
import torch.utils as utils

n_classes = 10

device = None
if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


class Dataset(utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)


def show(arr):
    plt.imshow(arr, cmap="Greys")
    plt.show()


def norm(X) -> torch.Tensor:
    """Normalise the dataset and return tensor."""
    x = torch.from_numpy(X)
    for im in x:  # remove noise from images, probably needs some work
        im[im < 220] = 0

    # add artificial 4th dimension to fit dimensions needed for conv layer
    return x.unsqueeze(1)
