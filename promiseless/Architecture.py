from promiseless.initialization import RandomInitialization
from promiseless.loss import MSE
from promiseless.Model import Model


class Architecture:
    def __init__(self):
        self._input_layer = None
        self._layers = []
        self._init_method = RandomInitialization()
        self._loss_function = MSE()

    def add_input_layer(self, input_layer):
        self._input_layer = input_layer
        return self

    def add_layer(self, layer):
        self._layers.append(layer)
        return self

    def set_initialization_method(self, initialization_method):
        self._init_method = initialization_method

    def set_loss_function(self, loss_function):
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
