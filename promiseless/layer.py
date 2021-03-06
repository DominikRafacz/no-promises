import numpy as np
import copy
from promiseless.activation import LinearActivation


class InputLayer:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class HiddenLayer:
    def __init__(self, size, activation=LinearActivation(), bias=True):
        self._out_size = size
        self._activation = activation
        self._bias = bias
        self._weights = None
        self._in_size = None

    def build(self, in_size, initialization_method):
        ret = copy.deepcopy(self)
        ret._in_size = in_size
        ret._weights = initialization_method.perform((
            ret._in_size if not ret._bias else ret._in_size + 1,
            ret._out_size
        ))
        return ret

    def out_size(self):
        return self._out_size

    def feedforward(self, data):
        # add vector of ones to the data
        if self._bias:
            data = np.concatenate((
                np.ones((data.shape[0], 1)),
                data
            ))

        # TODO: add applying of activation
        return data * self._weights

    def backpropagate(self, data):
        pass