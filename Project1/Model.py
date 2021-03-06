import numpy as np
import pandas as pd


class Model:

    def __init__(self, input_layer, layers, loss_function):
        self.input_layer = input_layer
        self.layers = layers
        self.loss_function = loss_function
        self.training_history = np.empty()

    def train(self, x_train, y_train, bias, batch_size, epochs, learning_rate, momentum, evaluation_dataset=None):
        for i, x in enumerate(x_train):
            for j, layer in enumerate(self.layers):
                layer.feedforward(x)
            loss = self.loss_function.calculate(x, y_train)
            error = self.loss_function.derivative(x, y_train)

    def predict(self, x_test, y_test):
        pass


name = "data.simple.test.100"
data = pd.read_csv("C:\\Users\\wojte\\OneDrive\\Dokumenty\\Studia\\Deep Learning\\projekt1\\classification\\{}.csv".format(name))
x_train = data.loc[:, ["x","y"]]


def create_batches(df, batch_size):
    n = len(df)
    new_df = data.iloc[np.random.permutation(n)]
    batches = [new_df.iloc[]]