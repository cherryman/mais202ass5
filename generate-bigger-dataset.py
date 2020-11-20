from utils import generate_transformed_image
import pandas as pd 
import numpy as np 
import random 

# Load dataset
X_train = np.load("train_x.npy")
y_train = pd.read_csv("train_y.csv", index_col="ID").to_numpy().reshape((-1))

# Get number of each class
_, counts = np.unique(y_train, return_counts=True)

# Sort dataset to know where to get base images for each class from
X_train = [x for _, x in sorted(zip(y_train,X_train), key=lambda pair: pair[0])] 
y_train = sorted(y_train)
# Maybe check to make sure this worked 

# Denoise data
for im in X_train:
    im[im > 220] = 0
# And check to make sure this worked

NUM_PER_CLASS = 15000   # I have 16 GB of RAM and 20000 per class was too much for my computer to handle, could probably be fixed
                        # by separating into smaller batches but I was too lazy

# Generate new images for each class
new_images = []
start_index = 0
end_index = counts[0]
for num in range(len(counts)):
    i = 0
    print("Generating images of " + str(num))
    print("Start", start_index)
    print("End", end_index)
    
    while i < (NUM_PER_CLASS - counts[num]):
        random_index = random.randrange(start_index, end_index)
        new_images.append(generate_transformed_image(X_train[random_index]))
        
        y_train = np.append(y_train, num)
        i += 1
    if num != len(counts) - 1:
        start_index = end_index
        end_index = end_index + counts[num + 1]
    
print(counts)
X_train = np.concatenate((X_train,np.array(new_images)),axis=0)

np.save(f"{NUM_PER_CLASS}-per-class-sorted-x-train.npy", X_train)
np.save("{NUM_PER_CLASS}-per-class-sorted-y-train.npy", y_train)