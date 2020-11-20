import matplotlib.pyplot as plt
import torch
import torch.utils as utils
import numpy as np
import torch.nn as nn
import random
import pandas as pd
from PIL import Image, ImageOps
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

    for im in x:
        im[im < 255] = 0

    # add artificial 4th dimension to fit dimensions needed for conv layer
    return x.unsqueeze(1)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)

    
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)

def add_noise(image):
    row,col= image.shape
    mean = np.mean(image)
    var = 100
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy

# print(counts)
def generate_transformed_image(im):
    im = Image.fromarray(im)
    x_translation = random.randrange(-12,12)
    y_translation = random.randrange(-12,12)
    im = im.transform(im.size, Image.AFFINE, (1,0,x_translation,0,1,y_translation))
    
    rotation_angle = random.randrange(-15,15)
    im = im.rotate(rotation_angle)
    im = np.array(im)
    for x in range(len(im)):
        for y in range(len(im[0])):
            if random.randrange(0,100) > 95:
                im[x,y] = random.randrange(220,255)
    
    return im



