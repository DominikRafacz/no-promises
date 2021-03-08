import numpy as np
from layer import InputLayer, HiddenLayer
from loss import LossFunction


class Model:
    def __init__(self, input_layer: InputLayer, layers: list[HiddenLayer], loss_function: LossFunction):
        self._input_layer = input_layer
        self._layers = layers
        self._loss_function = loss_function
        self._training_history = np.empty(0)

    @staticmethod
    def __create_batches(df, batch_size):
        '''
        :param df: numpy array
        :param batch_size: size of the batches
        :return: list of numpy arrays
        '''
        n = len(df)
        new_df = df[np.random.permutation(n)]
        batches = [new_df[start:start + batch_size] for start in range(0, n, batch_size)]
        return batches

    def train(self, x_train: np.ndarray, y_train: np.ndarray, bias, batch_size, epochs, learning_rate, momentum, evaluation_dataset=None):
        batches = self.__create_batches(x_train, batch_size)
        for batch in batches:
            for i, x in enumerate(batch):
                data = x
                for j, layer in enumerate(self._layers):
                    data = layer.feedforward(data)
                loss = self._loss_function.calculate(data, y_train[i])
                error = self._loss_function.derivative(data, y_train[i])

    def predict(self, x_test, y_test):
        pass

