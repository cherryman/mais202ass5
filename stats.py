import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

y_train = pd.read_csv("train_y.csv", index_col="ID").to_numpy().reshape((-1))

unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts)
plt.show()
