import numpy as np
import pandas as pd
from promiseless.Architecture import Architecture
from promiseless.layer import InputLayer, HiddenLayer


architecture = Architecture()\
    .add_input_layer(InputLayer(23))\
    .add_layer(HiddenLayer(456))\
    .add_layer(HiddenLayer(3))

model = architecture.build_model()


name = "data.simple.test.100"
data = pd.read_csv("C:\\Users\\wojte\\OneDrive\\Dokumenty\\Studia\\Deep Learning\\projekt1\\classification\\{}.csv".format(name))
x_train = data.loc[:, ["x","y"]]
x_train = np.array(x_train)
y_train = np.array(data.loc[:, "cls"])

def create_batches(x_train, y_train, batch_size):
    '''
    :param x_train: data in numpy array
    :param y_train: target in numpy array
    :param batch_size: size of batches
    :return: list of data batches and target batches
    '''
    n = len(x_train)
    perm = np.random.permutation(n)
    new_x = x_train[perm]
    new_y = y_train[perm]
    batches_x = [new_x[start:start + batch_size] for start in range(0, n, batch_size)]
    batches_y = [new_y[start:start + batch_size] for start in range(0, n, batch_size)]
    return batches_x, batches_y


batches_x, batches_y = create_batches(x_train, y_train, 10)