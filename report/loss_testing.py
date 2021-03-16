import numpy as np
from promiseless.architecture import Architecture
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.activation import Tanh, LinearActivation, Softmax
from promiseless.util import read_data, visualize_losses
from promiseless.loss import MAE, CategoricalCrossEntropy

x_train, y_train = read_data("regression", "data.cube.train.10000")

np.random.seed(2137)
mdl_regr_1 = Architecture() \
    .add_input_layer(InputLayer(1)) \
    .add_layer(HiddenLayer(10, activation=Tanh)) \
    .add_layer(HiddenLayer(10, activation=Tanh))\
    .add_layer(HiddenLayer(1, activation=LinearActivation)) \
    .build_model()
mdl_regr_2 = Architecture() \
    .add_input_layer(InputLayer(1)) \
    .add_layer(HiddenLayer(10, activation=Tanh)) \
    .add_layer(HiddenLayer(10, activation=Tanh))\
    .add_layer(HiddenLayer(1, activation=LinearActivation)) \
    .set_loss_function(MAE) \
    .build_model()

mdl_regr_1.train(x_train, y_train, 1000, 100, 0.001, momentum_lambda=0.9)
mdl_regr_2.train(x_train, y_train, 1000, 100, 0.001, momentum_lambda=0.9)

visualize_losses([mdl_regr_1, mdl_regr_2], ["MSE", "MAE"], filename="losses-regr")

x_train, y_train = read_data("classification", "data.simple.train.10000")

np.random.seed(6969)
mdl_classif_1 = Architecture() \
    .add_input_layer(InputLayer(2)) \
    .add_layer(HiddenLayer(10, activation=Tanh))\
    .add_layer(HiddenLayer(2, activation=Softmax)) \
    .build_model()
mdl_classif_2 = Architecture() \
    .add_input_layer(InputLayer(2)) \
    .add_layer(HiddenLayer(10, activation=Tanh))\
    .add_layer(HiddenLayer(2, activation=Softmax)) \
    .set_loss_function(CategoricalCrossEntropy) \
    .build_model()

mdl_classif_1.train(x_train, y_train, 100, 100, 0.001, momentum_lambda=0.9)
mdl_classif_2.train(x_train, y_train, 100, 100, 0.001, momentum_lambda=0.9)

visualize_losses([mdl_classif_1, mdl_classif_2], ["MSE", "CrossEntropy"], filename="losses-classif")