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
    for im in x:  # remove noise from images, probably needs some work
        im[im < 220] = 0

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

# X_train = np.load("sorted_X_train.npy")

# y_train = np.load("sorted_y_train.npy")
# counts = np.load("counts.npy")
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
# new_images = []
# start_index = 0
# end_index = counts[0]
# for num in range(len(counts) - 1):
#     i = 0
#     print("Start", start_index)
#     print("End", end_index)
#     while i < (10000-counts[num]):
#         random_index = random.randrange(start_index, end_index)
#         new_images.append(generate_transformed_image(X_train[random_index]))
        
#         y_train = np.append(y_train, num)
#         i += 1
#     start_index = end_index
#     end_index = end_index + counts[num + 1]
    

# X_train = np.concatenate((X_train,np.array(new_images)),axis=0)

# np.save("balanced_sorted_x_train.npy", X_train)
# np.save("balanced_sorted_y_train.npy", y_train)
# for im in X_train:
#     show(im)

# for index, im in enumerate(X_train):
#     print(y_train[index])
#     show(im)

