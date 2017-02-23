import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
print("Hello Logistic Regression")

from util import create_dataset 

x,y = create_dataset(100, insert_x0=False)
learning_rate = 0.01
n = len(x)
w = np.array([0,0])
def error(x,y,w):
    n = len(x)
    err = 0
    for i in range(0,n):
        err += np.log(1 + np.exp(np.dot(-y[i]*w.T,x[i])))
    err = err / n
    return err

err = err = error(x,y,w)
iter = 0
max_iter = 1000
while err > 0.01 and iter < max_iter:
    iter += 1
    d_err = 0
    for i in range(0, n):
        d_err += y[i]*x[i] / (1 + np.exp(y[i]*np.dot(np.transpose(w),x[i])))
    d_err /= -n
    w = w - learning_rate * d_err
    err = error(x,y,w)

print(w)
_, axis = plt.subplots()

step = 0.05
x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
x_mesh, y_mesh = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
x_mesh_flattened = x_mesh.flatten()
y_mesh_flattened = y_mesh.flatten()
cfdata = np.transpose(np.array((x_mesh_flattened, y_mesh_flattened)))
classification = ((np.sign(np.dot(cfdata, w))+1)/2).reshape(x_mesh.shape)
# draw boundaries
axis.contourf(x_mesh, y_mesh, classification, cmap=plt.cm.Paired)
# draw points
axis.scatter(x[:, 0], x[:, 1], marker='x', c=y)
axis.set_title('Logistic Regression')
axis.axis('off')
axis.set_aspect('equal', 'datalim')
plt.show()

