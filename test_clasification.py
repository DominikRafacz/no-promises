import numpy as np
from promiseless.architecture import Architecture
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.activation import Tanh, LinearActivation, Softmax
from promiseless.util import read_data, visualize_loss, visualize_results
from promiseless.loss import CategoricalCrossEntropy
import matplotlib.pyplot as plt

x, y = read_data("classification", "data.circles.train.1000")
x_test, y_test = read_data("classification", "data.circles.test.1000")

np.random.seed(2137)
mdl = (Architecture()
       .add_input_layer(InputLayer(2))
       .add_layer(HiddenLayer(4, activation=Softmax))
       .set_loss_function(CategoricalCrossEntropy)
       .build_model())

mdl.train(x, y, 100, 1000, momentum_lambda=0.9, learning_rate=0.001)
res, _ = mdl.predict(x, y, return_class=True)

visualize_results(x, res, y, "classification")