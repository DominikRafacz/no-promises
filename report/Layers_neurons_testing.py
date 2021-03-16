import numpy as np
from promiseless.architecture import Architecture
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.activation import Tanh, LinearActivation
from promiseless.util import read_data, visualize_losses

x_train, y_train = read_data("regression", "data.cube.train.1000")
x_test, y_test = read_data("regression", "data.cube.test.1000")

hidden_layers = [1, 2, 3, 5]
layer_sizes = [2, 5, 10]

models = []
labels = []
res = [None]*12
loss = [0]*12
i = 0

np.random.seed(1234)
for hidden_layer in hidden_layers:
    for layer_size in layer_sizes:
        mdl = Architecture().add_input_layer(InputLayer(1))
        for _ in range(hidden_layer):
            mdl = mdl.add_layer(HiddenLayer(layer_size, activation=Tanh))
        mdl = mdl.add_layer(HiddenLayer(1, activation=LinearActivation))
        mdl = mdl.build_model()
        mdl.train(x_train, y_train, 100, 500, 0.001, evaluation_dataset=(x_test, y_test), momentum_lambda=0.9)
        res[i], loss[i] = mdl.predict(x_test, y_test)
        models.append(mdl)
        labels.append(("{}-".format(layer_size)*hidden_layer)[:-1])
        i += 1

visualize_losses(models, labels=labels, data="train", start_from=10,
                 styles=({'color': color, 'linestyle': linestyle} for linestyle in ['solid', 'dotted', 'dashed', 'dashdot'] for color in ['red', 'green', 'blue'] ),
                 filename='sizes-comparison')

