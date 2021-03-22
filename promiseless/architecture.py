from typing import Type, Union
from promiseless.initialization import RandomInitialization, InitializationMethod
from promiseless.loss import MSE, LossFunction
from promiseless.model import Model
from promiseless.layer import InputLayer, HiddenLayer


class Architecture:
    def __init__(self):
        self._input_layer: Union[InputLayer, None] = None
        self._layers: list[HiddenLayer] = []
        self._init_method: Type[InitializationMethod] = RandomInitialization
        self._loss_function: Type[LossFunction] = MSE

    def add_input_layer(self, input_layer: InputLayer):
        self._input_layer = input_layer
        return self

    def add_layer(self, layer: HiddenLayer):
        self._layers.append(layer)
        return self

    def set_initialization_method(self, initialization_method: Type[InitializationMethod]):
        self._init_method = initialization_method
        return self

    def set_loss_function(self, loss_function: Type[LossFunction]):
        self._loss_function = loss_function
        return self

    def build_model(self):
        built_layers = [self._layers[0].build(self._input_layer.size(), self._init_method)]
        previous_out_size = built_layers[0].out_size()

        for layer in self._layers[1:]:
            next_layer = layer.build(previous_out_size, self._init_method)
            built_layers.append(next_layer)
            previous_out_size = next_layer.out_size()

        return Model(self._input_layer, built_layers, self._loss_function)
