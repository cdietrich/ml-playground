#! python3
'''
Perceptron Learning Alogrithm Example
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from util import create_dataset 
import sklearn
print("Hello Perceptron")


def calculate_misses(weights, features, labels):
    """
    calculates missclassified indices in given x

    Parameters
    ----------
    weights :
        weight vector
    features :
        feature matrix incl. x0 = 1
    labels :
        labels normalized to +-1
    Returns
    -------
    res : list
        List of missclassified indices.
    """
    res = []
    # go through the training set
    for i in range(1, len(features)):
        y_predict = np.sign(np.dot(weights.T, features[i]))
        # collect missclassifed indices
        if y_predict != labels[i]:
            res.append(i)
    return res

def train(x_train, y_train, max_iter=1000):
    """
    trains on the given data.

    Parameters
    ----------
    x_train:
        feature matrix (incl. x0 = 1)
    y_train:
        label vector (normalized to +-1)
    max_iter:
        max number of iterations
    Returns
    -------
    weights
    misses
    """
    # init w
    number_of_features = len(x_train[0])
    weights = np.random.rand(number_of_features)
    done = False
    iter = 0
    while not done and iter < max_iter:
        iter += 1
        misses = calculate_misses(weights, x_train, y_train)
        if len(misses) == 0:
            done = True
        else:
            pick = np.random.choice(misses)
            update = y_train[pick]*x_train[pick]
            weights = weights + update
    return weights, misses

def contour_data(features, weights):
    """
    calculates contour data based on given train features ranges and weights

    Parameters
    ----------
    features :
        feature matrix incl. x0 = 1
    weights :
        weight vector
    Returns
    -------
    x_mesh :
    y_mesh :
    classification :
    """
    step = 0.01
    x1_min, x1_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    x2_min, x2_max = features[:, 2].min() - 1, features[:, 2].max() + 1
    x_mesh, y_mesh = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
    x_mesh_flattened = x_mesh.flatten()
    y_mesh_flattened = y_mesh.flatten()
    cfdata = np.insert(np.transpose(np.array((x_mesh_flattened, y_mesh_flattened))), [0], 1, axis=1)
    classification = ((np.sign(np.dot(cfdata, weights))+1)/2).reshape(x_mesh.shape)
    return x_mesh, y_mesh, classification

def train_and_show(x_train, y_train):
    """
    trains on given data, shows result as scatter + contour plot
    """
    # do the training
    weights, _ = train(x_train, y_train)
    # calculate boundaries
    x_mesh, y_mesh, classification = contour_data(x_train, weights)
    # draw it
    _, axis = plt.subplots()
    # draw boundaries
    axis.contourf(x_mesh, y_mesh, classification, cmap=plt.cm.Paired)
    # draw points
    axis.scatter(x_train[:, 1], x_train[:, 2], marker='x', c=y_train)
    axis.set_title('Perceptron')
    axis.axis('off')
    plt.show()

def main():
    """
    Main method.
    """
    # create the dataset
    x_train, y_train = create_dataset()
    train_and_show(x_train, y_train)

if __name__ == "__main__":
    # execute only if run as a script
    main()
