import numpy as np
from promiseless.architecture import Architecture
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.activation import Sigmoid, ReLU, Tanh, Softmax, LinearActivation
from promiseless.loss import CategoricalCrossEntropy, MSE, MAE
from promiseless.util import read_data, visualize_loss, visualize_results, visualize_losses, visualize_results2

name = "data.cube.train.100"
x_train, y_train = read_data("regression", name)
x_test, y_test = read_data("regression", "data.cube.test.100")


name2 = "data.three_gauss.train.10000"
x_train2, y_train2 = read_data("classification", name2)
name2 = "data.three_gauss.test.10000"
x_test2, y_test2 = read_data("classification", name2)


activation = Sigmoid
hidden_layers = [1, 2, 3, 5]
loss_functions = [MSE, MAE]
layer_sizes = [2, 5, 10]

models = []
labels = []
res = [None]*12
loss = [0]*12
i = 0
for hidden_layer in hidden_layers:
    for layer_size in layer_sizes:
        np.random.seed(1234)
        mdl = Architecture().add_input_layer(InputLayer(1))
        for _ in range(hidden_layer):
            mdl = mdl.add_layer(HiddenLayer(layer_size, activation=activation))
        mdl = mdl.add_layer(HiddenLayer(1, activation=LinearActivation))
        mdl = mdl.build_model()
        mdl.train(x_train, y_train, 5, 1000, 0.001, evaluation_dataset=(x_test, y_test), momentum_lambda=0.9)
        res[i], loss[i] = mdl.predict(x_test, y_test)
        models.append(mdl)
        labels.append(("{}-".format(layer_size)*hidden_layer)[:-1])
        i += 1

visualize_losses(models, labels=labels, data="train", start_from=10)
visualize_losses(models, labels=["LinearActivation", "Tanh", "ReLU", "Sigmoid"], data="test")
visualize_results2(x_test, res, y_test, "regression", labels=labels)
