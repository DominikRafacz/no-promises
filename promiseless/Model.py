import numpy as np
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.loss import LossFunction


class Model:
    def __init__(self, input_layer: InputLayer, layers: list[HiddenLayer], loss_function: LossFunction):
        self._input_layer = input_layer
        self._layers = layers
        self._loss_function = loss_function
        self._training_history = np.empty(0)

    @staticmethod
    def __create_batches(x_train, y_train, batch_size):
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

    def train(self, x_train: np.ndarray, y_train: np.ndarray, bias, batch_size, epochs, learning_rate, momentum, evaluation_dataset=None):
        for _ in range(epochs):
            batches_x, batches_y = self.__create_batches(x_train, y_train, batch_size)
            for h, x_train_batch in enumerate(batches_x):
                y_train_batch = batches_y[h]
                for i, x in enumerate(x_train_batch):
                    data = x
                    for j, layer in enumerate(self._layers):
                        data = layer.feedforward(data)
                    loss = self._loss_function.calculate(data, y_train_batch[i])
                    error = self._loss_function.derivative(data, y_train_batch[i])
                    for j, layer in reversed(list(enumerate(self._layers))):
                        error = layer.backpropagate(error)

    def predict(self, x_test, y_test):
        for i, x in enumerate(x_test):
            data = x
            for j, layer in enumerate(self._layers):
                data = layer.feedforward(data)
            loss = self._loss_function.calculate(data, y_test[i])
        return data, loss

