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

def create_batches(df, batch_size):
    '''
    :param df: numpy array
    :param batch_size: size of the batches
    :return: list of numpy arrays
    '''
    n = len(df)
    new_df = df[np.random.permutation(n)]
    batches = [new_df[start:start + batch_size] for start in range(0, n, batch_size)]
    return batches


batches = create_batches(x_train, 1)