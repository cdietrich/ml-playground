import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.optimize
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import time
mnist = read_data_sets("MNIST_data/", one_hot=False)

def get_ground_truth(labels):
    labels = np.array(labels).flatten()
    k = len(labels)
    data = np.ones(k)
    indptr = np.arange(k + 1)
    ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
    ground_truth = ground_truth.todense().T
    return ground_truth

def softmax_cost(theta, features, labels, lam=0.01):

    number_of_features = features.shape[1]
    ground_truth = get_ground_truth(labels)

    theta_x = np.dot(theta.T, features)
    probabilities = softmax(theta_x)

    cost_examples = np.multiply(ground_truth, np.log(probabilities))
    traditional_cost = -(np.sum(cost_examples) / number_of_features)

    theta_squared = np.multiply(theta, theta)
    weight_decay = lam / 2 * np.sum(theta_squared)
    cost = traditional_cost + weight_decay

    theta_grad = (-np.dot(ground_truth - probabilities, features.T)) / number_of_features + lam * theta.T
    theta_grad = np.array(theta_grad)

    return [cost, theta_grad.T]

def softmax_predict(theta, features):

    number_of_features = features.shape[1]
    theta_x = np.dot(theta.T, features)
    probabilities = softmax(theta_x)

    predictions = np.zeros(shape=(number_of_features, 1))
    predictions[:, 0] = np.argmax(probabilities, axis=0)
    return predictions

def softmax(values):
    values = values - np.max(values)
    hypothesis = np.exp(values)
    probabilities = hypothesis / np.sum(hypothesis, axis=0)
    return probabilities

def train(number_of_classes, data, batch_size=1000, iterations=1000, lam=0, learning_rate=0.2):
    features, labels = data.next_batch(batch_size)
    number_of_features = features.shape[1]
    weights = np.zeros(shape=(number_of_features, number_of_classes))
    costs = []
    for i in range(0, iterations):
        cost, grad2 = softmax_cost(weights, features.T, labels, lam)
        costs.append(cost)
        weights = weights - (learning_rate * grad2)
        features, labels = mnist.train.next_batch(batch_size)
    return weights, costs

def accuracy(weights, features, labels):
    predict = softmax_predict(weights, features)
    correct = []
    i = 0
    for label in labels:
        if label == predict[i, 0]:
            correct.append(1)
        else:
            correct.append(0)
        i = i + 1
    return np.mean(correct)

weights, costs = train(10, mnist.train, batch_size=1000, iterations=1000, lam=0, learning_rate=0.2)
plt.plot(costs)
plt.show()
acc = accuracy(weights, mnist.test.images.T, mnist.test.labels)
print("""Accuracy :""", acc)
