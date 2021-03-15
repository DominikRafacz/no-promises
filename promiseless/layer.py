from typing import Type

import copy
import numpy

from promiseless.activation import LinearActivation, ActivationFunction
from promiseless.initialization import InitializationMethod


class InputLayer:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class HiddenLayer:
    def __init__(self, size, activation: Type[ActivationFunction] = LinearActivation, bias: bool = True):
        self._out_size = size
        self._activation = activation
        self._bias = bias
        self._weights = None
        self._in_size = None
        self.shape = None

    def build(self, in_size: int, initialization_method: Type[InitializationMethod]):
        ret = copy.deepcopy(self)
        ret._in_size = in_size
        ret._weights = initialization_method.perform((
            ret._in_size if not ret._bias else ret._in_size + 1,
            ret._out_size
        ))
        ret._shape = ret._weights.shape
        return ret

    def out_size(self):
        return self._out_size

    def feedforward(self, data: numpy.ndarray):
        # add vector of ones to the data
        if self._bias:
            data = numpy.concatenate((
                numpy.ones((data.shape[0], 1)),
                data
            ), axis=1)
        val = data @ self._weights
        return val, self._activation.calculate(val)

    def calculate_delta_weights(self, error: numpy.ndarray, prev_values: numpy.ndarray):
        delta_weights = numpy.zeros(self._weights.shape)
        if self._bias:
            for i in range(error.shape[0]):
                delta_weights += numpy.concatenate((numpy.ones((1, 1)), prev_values[i].reshape(-1, 1))) @ error[i].reshape(1, -1)
        else:
            for i in range(error.shape[0]):
                delta_weights += prev_values[i].reshape(-1, 1) @ error[i].reshape(1, -1)
        return delta_weights

    def calculate_prev_error(self, prev_derivative: numpy.ndarray, error: numpy.ndarray):
        if self._bias:
            return error @ self._weights[1:, :].T * prev_derivative
        else:
            return error @ self._weights.T * prev_derivative

    def update_weights(self, delta_weights: numpy.ndarray, learning_rate: numpy.ndarray):
        self._weights -= delta_weights * learning_rate

    def derivative(self, data: numpy.ndarray):
        return self._activation.derivative(data)
