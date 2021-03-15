import numpy as np
from promiseless.architecture import Architecture
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.activation import Sigmoid, ReLU, Tanh, Softmax, LinearActivation
from promiseless.loss import CategoricalCrossEntropy, MSE, MAE
from promiseless.util import read_data, visualize_loss, visualize_results

name = "data.cube.train.100"
x_train, y_train = read_data("regression", name)
x_test, y_test = read_data("regression", "data.cube.test.100")


name2 = "data.simple.train.100"
x_train2, y_train2 = read_data("classification", name2)

np.random.seed(123)

mdl = Architecture()\
    .add_input_layer(InputLayer(1))\
    .add_layer(HiddenLayer(5, activation=Tanh))\
    .add_layer(HiddenLayer(1))\
    .build_model()


mdl.train(x_train, y_train, batch_size=10, learning_rate=10e-4, epochs=100, evaluation_dataset=(x_test, y_test))

res, loss = mdl.predict(x_test, y_test)
'''
task = "regression"
activations = [Tanh, ReLU, Sigmoid]
batches = [1, 10, 100, 500]
hidden_layers = [1, 2, 3, 5]
loss_functions = [MSE, MAE]
layer_sizes = [2, 5, 10]

for hidden_layer in hidden_layers:
    for activation in activations:
        for loss_function in loss_functions:
            for batch in batches:
                for layer_size in layer_sizes:
                    i = 0
                    np.random.seed(123)
                    mdl = Architecture().add_input_layer(InputLayer(1))
                    for _ in range(hidden_layer):
                        mdl = mdl.add_layer(HiddenLayer(layer_size, activation=activation))
                    mdl = mdl.set_loss_function(loss_function=loss_function).build_model()
                    mdl.train(x_train, y_train, batch_size=batch, learning_rate=10e-4, epochs=100)
                    res, loss = mdl.predict(x_test, y_test)
                    visualize_results(x_test, res, y_test, "regression", "test{}".format(i))
                    visualize_loss(mdl, "test{}".format(i))
                    i += 1

# np.random.seed(123)
#
# mdl_classif = Architecture()\
#     .add_input_layer(InputLayer(2))\
#     .add_layer(HiddenLayer(5, activation=Sigmoid))\
#     .add_layer(HiddenLayer(2, activation=Softmax))\
#     .set_loss_function(CategoricalCrossEntropy)\
#     .build_model()
#
# mdl_classif.train(x_train2, y_train2, batch_size=100, learning_rate=10e-4, epochs=100)
#
# res2, loss2 = mdl_classif.predict(x_train2, y_train2, return_class=True)



'''

visualize_results(x_test, res, y_test, "regression", "test2")
visualize_results(x_test, res, y_test, "regression")
# visualize_results(x_train2, res2, y_train2, "classification")





visualize_loss(mdl, "test")
#
# plt.scatter(x_test, res)
# plt.show()
# print(mdl.training_history)
#
# plt.plot(range(1,101),mdl.training_history[0])
# plt.plot(range(1,101),mdl.training_history[1])
