import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from promiseless.Architecture import Architecture
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.activation import Sigmoid


def read_data(task, dataset_name):
    data = pd.read_csv("data/{0}/{1}.csv".format(task, dataset_name))
    if task == "regression":
        x_train = data.loc[:, ["x"]]
        x_train = np.array(x_train)
        x_train = (x_train - x_train.mean()) / x_train.std()
        y_train = np.array(data.loc[:, ["y"]])
        y_train = (y_train - y_train.mean()) / y_train.std()
    elif task == "classification":
        x_train = data.loc[:, ["x", "y"]]
        x_train = np.array(x_train)
        x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
        y_train = np.array(data.loc[:, ["cls"]])
    else:
        print("Unknown task")
        x_train = data
        y_train = None
    return x_train, y_train


name = "data.cube.train.100"
x_train, y_train = read_data("regression", name)

name2 = "data.simple.train.100"
x_train2, y_train2 = read_data("classification", name2)

plt.scatter(x_train, y_train)
plt.show()

plt.scatter(x_train2[:, 0], x_train2[:, 1], c=y_train2.reshape(1, -1))
plt.show()

np.random.seed(123)

mdl = Architecture()\
    .add_input_layer(InputLayer(1))\
    .add_layer(HiddenLayer(5, activation=Sigmoid))\
    .add_layer(HiddenLayer(1))\
    .build_model()

loss1 = mdl.train(x_train, y_train, batch_size=10, learning_rate=10e-4, epochs=100)

res, _ = mdl.predict(x_train, y_train)
plt.scatter(x_train, res)
plt.show()

np.random.seed(123)
mdl2 = Architecture()\
    .add_input_layer(InputLayer(1))\
    .add_layer(HiddenLayer(5, activation=Sigmoid))\
    .add_layer(HiddenLayer(1))\
    .build_model()

loss2 = mdl2.train(x_train, y_train, batch_size=100, learning_rate=10e-4, epochs=50, momentum_lambda=0.9)
res2, _ = mdl2.predict(x_train, y_train)
plt.scatter(x_train, res2)
plt.show()

####
from plotnine import ggplot, aes, geom_point
from pandas import DataFrame

ggplot(DataFrame(np.concatenate((x_train, y_train), axis=1), columns=("x", "y"))) + aes(x="x", y="y") + geom_point()


plt.plot(range(10,101), loss1[9:], label="Without momentum")
plt.plot(range(10,51), loss2[9:], label="With momentum")
plt.title("Loss")
plt.legend()
plt.show()


