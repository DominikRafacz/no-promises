import numpy
from typing import Type, Tuple, Union
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.loss import LossFunction


class Model:
    def __init__(self, input_layer: InputLayer, layers: list[HiddenLayer], loss_function: Type[LossFunction]):
        self._input_layer = input_layer
        self._layers = layers
        self._loss_function = loss_function
        self._training_history = []

    @staticmethod
    def __create_batches(x_train, y_train, batch_size):
        '''
        :param x_train: data in numpy array
        :param y_train: target in numpy array
        :param batch_size: size of batches
        :return: list of data batches and target batches
        '''
        n = len(x_train)
        perm = numpy.random.permutation(n)
        new_x = x_train[perm]
        new_y = y_train[perm]
        batches_x = [new_x[start:start + batch_size] for start in range(0, n, batch_size)]
        batches_y = [new_y[start:start + batch_size] for start in range(0, n, batch_size)]
        return batches_x, batches_y

    def train(self, x_train: numpy.ndarray, y_train: numpy.ndarray, batch_size=1, epochs=10, learning_rate=0.01, momentum_lambda=0, evaluation_dataset: Union[Tuple[numpy.ndarray], None] = None):
        momentum = [numpy.zeros(layer.shape) for layer in self._layers]
        for ep in range(epochs):
            batches_x, batches_y = self.__create_batches(x_train, y_train, batch_size)
            for data_x, data_y in zip(batches_x, batches_y):
                layer_values = [None] * (len(self._layers) + 1)
                layer_values[0] = data_x
                for j, layer in enumerate(self._layers):
                    layer_values[j + 1], data_x = layer.feedforward(data_x)
                error = self._loss_function.derivative(data_x, data_y) * self._layers[-1].derivative(
                    layer_values[-1])
                for j, layer in reversed(list(enumerate(self._layers))):
                    delta_weights = layer.calculate_delta_weights(error, layer_values[j])
                    momentum[j] = momentum_lambda * momentum[j] + (1 - momentum_lambda) * delta_weights
                    if j != 0:
                        error = layer.calculate_prev_error(self._layers[j - 1].derivative(layer_values[j]), error)
                    layer.update_weights(momentum[j], learning_rate)
            train_loss = self.predict(x_train, y_train)[1]
            if evaluation_dataset:
                eval_loss = self.predict(evaluation_dataset[0], evaluation_dataset[1])[1]
                self._training_history.append(numpy.array([train_loss, eval_loss]))
            self._training_history.append(numpy.array([train_loss, None]))

    def predict(self, x_test: numpy.ndarray, y_test: numpy.ndarray):
        data = x_test
        for j, layer in enumerate(self._layers):
            _, data = layer.feedforward(data)
        loss = self._loss_function.calculate(data, y_test)
        return data, loss

