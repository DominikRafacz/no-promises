import numpy as np


class Model:
    def __init__(self, input_layer, layers, loss_function):
        self._input_layer = input_layer
        self._layers = layers
        self._loss_function = loss_function
        self._training_history = np.empty(0)

    def train(self, x_train, y_train, bias, batch_size, epochs, learning_rate, momentum, evaluation_dataset=None):
        for i, x in enumerate(x_train):
            for j, layer in enumerate(self._layers):
                layer.feedforward(x)
            loss = self._loss_function.calculate(x, y_train)
            error = self._loss_function.derivative(x, y_train)

    def predict(self, x_test, y_test):
        pass

