import numpy as np
import pandas as pd
import time
from promiseless.architecture import Architecture
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.activation import Tanh, LinearActivation, Softmax, Sigmoid, ReLU
from promiseless.util import read_data, visualize_loss, visualize_results, transform, accuracy
from promiseless.loss import CategoricalCrossEntropy
from promiseless.initialization import XavierInitialization
import matplotlib.pyplot as plt

data = pd.read_csv("data/classification/mnist.train.csv")
y = np.array(pd.get_dummies(data.loc[:, "label"]))
x = np.array(data.iloc[:, 1:])
# x, x_mean, x_std = transform(x)
x = (x - x.mean())/x.std()


# data_test = pd.read_csv("data/classification/mnist.test.csv")
# x_test = np.array(data_test)
# x_test = transform(x_test, x_mean, x_std)

np.random.seed(69)
idx = np.random.choice(np.arange(0, 42000), size=42000, replace=False)
x_train = x[idx[:40000], :]
y_train = y[idx[:40000], :]
x_valid = x[idx[40000:], :]
y_valid = y[idx[40000:], :]

np.random.seed(2137)
mdl = (Architecture()
       .add_input_layer(InputLayer(784))
       .add_layer(HiddenLayer(100, activation=Sigmoid))
       .add_layer(HiddenLayer(10, activation=Softmax))
       .set_loss_function(CategoricalCrossEntropy)
       .build_model())

t1 = time.time()
mdl.train(x_train, y_train, batch_size=400, epochs=200, learning_rate=0.001, momentum_lambda=0.9,
          evaluation_dataset=(x_valid, y_valid))
t2 = time.time()
print("Time: {0:.4f}".format(t2 - t1))
visualize_loss(mdl)
res, loss = mdl.predict(x_valid, y_valid, return_class=True)

np.random.seed(2137)
mdl2 = (Architecture()
        .add_input_layer(InputLayer(784))
        .add_layer(HiddenLayer(256, activation=Sigmoid))
        .add_layer(HiddenLayer(128, activation=Sigmoid))
        .add_layer(HiddenLayer(64, activation=Sigmoid))
        .add_layer(HiddenLayer(10, activation=Softmax))
        .set_loss_function(CategoricalCrossEntropy)
        .set_initialization_method(XavierInitialization)
        .build_model())

t1 = time.time()
mdl2.train(x_train, y_train, batch_size=50, epochs=5, learning_rate=0.001, momentum_lambda=0.9,
           evaluation_dataset=(x_valid, y_valid))
t2 = time.time()
print("Time: {0:.4f}".format(t2 - t1))
visualize_loss(mdl2)

accuracy(y_valid, res)
pd.get_dummies(pd.Series(res.reshape(-1))).sum(axis=0)
