import numpy as np
import pandas as pd
from promiseless.architecture import Architecture
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.activation import Tanh, LinearActivation, Softmax, Sigmoid
from promiseless.util import read_data, visualize_loss, visualize_results
from promiseless.loss import CategoricalCrossEntropy
import matplotlib.pyplot as plt

data = pd.read_csv("data/classification/mnist.train.csv")
y = np.array(pd.get_dummies(data.loc[:, "label"]))
x = np.array(data.iloc[:, 1:])


def transform(x, x_mean=None, x_std=None):
       if x_mean is None:
              x_mean = x.mean(axis=0)
              x_std = x.std(axis=0)
       x_std_nonzero = x_std > 1e-4
       x = x - x_mean
       x[:, x_std_nonzero] = x[:, x_std_nonzero] / x_std[x_std_nonzero]
       return x, x_mean, x_std

x, x_mean, x_std = transform(x)

data_test = pd.read_csv("data/classification/mnist.test.csv")
x_test = np.array(data_test)
x_test = transform(x_test, x_mean, x_std)

np.random.seed(69)
idx = np.random.choice(np.arange(0, 42000), size=10000, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]

np.random.seed(2137)
mdl = (Architecture()
       .add_input_layer(InputLayer(784))
       .add_layer(HiddenLayer(32, activation=Sigmoid))
       .add_layer(HiddenLayer(32, activation=Sigmoid))
       .add_layer(HiddenLayer(10, activation=Softmax))
       .set_loss_function(CategoricalCrossEntropy)
       .build_model())

mdl.train(x_train, y_train, batch_size=64, epochs=100, learning_rate=0.0001)

res, loss = mdl.predict(x_train, y_train, return_class=True)


def accuracy(y, y_hat):
       return np.sum(y.argmax(axis=1).reshape(-1,1) == y_hat)/len(y)

accuracy(y_train, res)
pd.get_dummies(pd.Series(res.reshape(-1))).sum(axis=0)