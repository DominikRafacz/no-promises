import numpy as np
from promiseless.architecture import Architecture
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.activation import Tanh, LinearActivation, Softmax, Sigmoid
from promiseless.util import read_data, visualize_loss, visualize_results
import matplotlib.pyplot as plt

x, y = read_data("regression", "data.square.train.1000")
x_test, y_test = read_data("regression", "data.square.test.1000")

np.random.seed(2137)
mdl = (Architecture()
       .add_input_layer(InputLayer(1))
       .add_layer(HiddenLayer(10, activation=Sigmoid))
       .add_layer(HiddenLayer(1, activation=LinearActivation))
       .build_model())
mdl.train(x, y, 50, 100, learning_rate=0.01)
res, loss = mdl.predict(x, y)
res_test, loss_test = mdl.predict(x_test, y_test)


visualize_loss(mdl)
plt.scatter(x, res, label="Fitted values")
plt.scatter(x, y, label="Original values")
plt.title("Fitted vs original")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()