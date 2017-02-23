import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

def create_dataset(n_samples=100, insert_x0=True):
    """
    create a training dataset

    Parameters
    ----------
    n_samples:
        number of samples
    Returns
    -------
    x_train
    y_train
    """
    # create 2 class separable dataset using sklearn
    dataset = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.0,
                                     center_box=(-10.0, 10.0), shuffle=True)
    x_raw = dataset[0]
    y_raw = dataset[1]
    # add x0
    if (insert_x0):
        x_train = np.insert(x_raw, [0], 1, axis=1)
    else:
        x_train = x_raw
    # normalize y
    y_train = y_raw*2-1
    return x_train, y_train