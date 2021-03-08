import numpy as np
from layer import InputLayer, HiddenLayer
from loss import LossFunction


class Model:
    def __init__(self, input_layer: InputLayer, layers: list[HiddenLayer], loss_function: LossFunction):
        self._input_layer = input_layer
        self._layers = layers
        self._loss_function = loss_function
        self._training_history = np.empty(0)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, bias, batch_size, epochs, learning_rate, momentum, evaluation_dataset=None):
        for i, x in enumerate(x_train):
            data = x
            for j, layer in enumerate(self._layers):
                data = layer.feedforward(data)
            loss = self._loss_function.calculate(data, y_train[i])
            error = self._loss_function.derivative(data, y_train[i])

    def predict(self, x_test, y_test):
        pass

