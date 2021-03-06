import numpy as np
import pandas as pd


class Model:

    def __init__(self, input_layer, layers, loss_function):
        self.input_layer = input_layer
        self.layers = layers
        self.loss_function = loss_function
        self.training_history = np.empty()

    def train(self, x_train, y_train, bias, batch_size, epochs, learning_rate, momentum, evaluation_dataset=None):
        pass

    def predict(self, x_test, y_test):
        pass