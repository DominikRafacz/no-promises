import numpy as np
import pandas as pd
from promiseless.Architecture import Architecture
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.activation import Sigmoid


name = "data.activation.test.100"
data = pd.read_csv("~/Desktop/projekt1/regression/{}.csv".format(name))
x_train = data.loc[:, ["x"]]
x_train = np.array(x_train)
x_train = (x_train - x_train.mean()) / x_train.std()
y_train = np.array(data.loc[:, ["y"]])
y_train = (y_train - y_train.mean()) / y_train.std()

np.random.seed(123)

mdl = Architecture()\
    .add_input_layer(InputLayer(1))\
    .add_layer(HiddenLayer(5, activation=Sigmoid))\
    .add_layer(HiddenLayer(1))\
    .build_model()

mdl.train(x_train, y_train, batch_size=10000, learning_rate=10e-4, epochs=100)

####
from plotnine import ggplot, aes, geom_point
from pandas import DataFrame

ggplot(DataFrame(np.concatenate((x_train, y_train), axis=1), columns=("x", "y"))) + aes(x="x", y="y") + geom_point()
