import pandas as pd
import numpy as np
from promiseless.architecture import Architecture
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.activation import Sigmoid, ReLU, Tanh, Softmax, LinearActivation
from promiseless.loss import CategoricalCrossEntropy
from promiseless.util import read_data, visualize_loss, visualize_results
from plotnine import ggplot, geom_line, facet_wrap, aes, scale_size_manual, theme_minimal, labs, scale_x_continuous, geom_point

x_train, y_train = read_data("regression", "data.activation.train.100")
x_test, y_test = read_data("regression", "data.activation.test.100")

results = pd.DataFrame({"activation": "none",
                        "x": x_test[:, 0],
                        "y": y_test[:, 0],
                        "type": "real"})
losses = pd.DataFrame(columns=("activation", "type", "loss", "epoch"))

names = {LinearActivation: "Linear",
         Tanh: "Tanh",
         ReLU: "ReLU",
         Sigmoid: "Sigmoid"}

np.random.seed(1010)
for activation in [LinearActivation, Tanh, ReLU, Sigmoid]:
    mdl = Architecture()\
        .add_input_layer(InputLayer(1))\
        .add_layer(HiddenLayer(5, activation=activation))\
        .add_layer(HiddenLayer(5, activation=activation))\
        .add_layer(HiddenLayer(1, activation=LinearActivation))\
        .build_model()
    mdl.train(x_train, y_train, 5, 50, 0.001, evaluation_dataset=(x_test, y_test))
    res, loss = mdl.predict(x_test, y_test)
    results = pd.concat((results,
                         pd.DataFrame({"activation": names[activation],
                                       "x": x_test[:, 0],
                                       "y": res[:, 0],
                                       "type": "prediction"})
                         ))
    losses = pd.concat((losses,
                        pd.DataFrame({"activation": names[activation],
                                      "type": "train",
                                      "loss": mdl.training_history[0],
                                      "epoch": [float(i) for i in range(len(mdl.training_history[0]))]}),
                        pd.DataFrame({"activation": names[activation],
                                      "type": "test",
                                      "loss": mdl.training_history[1],
                                      "epoch": [float(i) for i in range(len(mdl.training_history[0]))]})
                        ))

ggplot(aes(x="x",
           y="y",
           group="activation",
           color="activation",
           size="type"), data=results) + \
    geom_line() + \
    labs(title="Comparison of fits") + \
    scale_size_manual(values=(0.5, 1, 1, 1, 1))

ggplot(aes(x="epoch",
           y="loss",
           group="type",
           color="type"), data=losses) + \
    geom_line() + \
    labs(title="Convergence comparison") + \
    facet_wrap("activation") + \
    scale_x_continuous()

############## another dataset

x_train, y_train = read_data("regression", "data.cube.train.100")
x_test, y_test = read_data("regression", "data.cube.test.100")

results = pd.DataFrame({"activation": "none",
                        "x": x_test[:, 0],
                        "y": y_test[:, 0],
                        "type": "real"})
losses = pd.DataFrame(columns=("activation", "type", "loss", "epoch"))

names = {LinearActivation: "Linear",
         Tanh: "Tanh",
         ReLU: "ReLU",
         Sigmoid: "Sigmoid"}

np.random.seed(2020)
for activation in [LinearActivation, Tanh, ReLU, Sigmoid]:
    mdl = Architecture()\
        .add_input_layer(InputLayer(1))\
        .add_layer(HiddenLayer(5, activation=activation))\
        .add_layer(HiddenLayer(5, activation=activation))\
        .add_layer(HiddenLayer(1, activation=LinearActivation))\
        .build_model()
    mdl.train(x_train, y_train, 5, 1000, 0.001, evaluation_dataset=(x_test, y_test))
    res, loss = mdl.predict(x_test, y_test)
    results = pd.concat((results,
                         pd.DataFrame({"activation": names[activation],
                                       "x": x_test[:, 0],
                                       "y": res[:, 0],
                                       "type": "prediction"})
                         ))
    losses = pd.concat((losses,
                        pd.DataFrame({"activation": names[activation],
                                      "type": "train",
                                      "loss": mdl.training_history[0],
                                      "epoch": [float(i) for i in range(len(mdl.training_history[0]))]}),
                        pd.DataFrame({"activation": names[activation],
                                      "type": "test",
                                      "loss": mdl.training_history[1],
                                      "epoch": [float(i) for i in range(len(mdl.training_history[0]))]})
                        ))

ggplot(aes(x="x",
           y="y",
           group="activation",
           color="activation",
           size="type"), data=results) + \
    geom_line() + \
    labs(title="Comparison of fits") + \
    scale_size_manual(values=(0.5, 1, 1, 1, 1))


ggplot(aes(x="epoch",
           y="loss",
           group="type",
           color="type"), data=losses) + \
    geom_line() + \
    labs(title="Convergence comparison") + \
    facet_wrap("activation") + \
    scale_x_continuous()

############## classification

x_train, y_train = read_data("classification", "data.simple.train.100")
x_test, y_test = read_data("classification", "data.simple.test.100")

results = pd.DataFrame(columns=("activation", "x", "y", "real_class", "predicted"))
losses = pd.DataFrame(columns=("activation", "type", "loss", "epoch"))

names = {LinearActivation: "Linear",
         Tanh: "Tanh",
         ReLU: "ReLU",
         Sigmoid: "Sigmoid"}

np.random.seed(3030)
for activation in [LinearActivation, Tanh, ReLU, Sigmoid]:
    mdl = Architecture()\
        .add_input_layer(InputLayer(2))\
        .add_layer(HiddenLayer(5, activation=activation))\
        .add_layer(HiddenLayer(5, activation=activation))\
        .add_layer(HiddenLayer(2, activation=Softmax))\
        .set_loss_function(CategoricalCrossEntropy)\
        .build_model()
    mdl.train(x_train, y_train, 5, 100, 0.001, evaluation_dataset=(x_test, y_test))
    res, loss = mdl.predict(x_test, y_test, return_class=True)
    results = pd.concat((results,
                         pd.DataFrame({"activation": names[activation],
                                       "x": x_test[:, 0],
                                       "y": x_test[:, 1],
                                       "real_class": y_test[:, 0],
                                       "predicted": res.reshape(-1)})
                         ))
    losses = pd.concat((losses,
                        pd.DataFrame({"activation": names[activation],
                                      "type": "train",
                                      "loss": mdl.training_history[0],
                                      "epoch": [float(i) for i in range(len(mdl.training_history[0]))]}),
                        pd.DataFrame({"activation": names[activation],
                                      "type": "test",
                                      "loss": mdl.training_history[1],
                                      "epoch": [float(i) for i in range(len(mdl.training_history[0]))]})
                        ))

ggplot(aes(x="x",
           y="y",
           color="predicted",
           shape="real_class"), data=results) + \
    geom_point() + \
    facet_wrap("activation")

ggplot(aes(x="epoch",
           y="loss",
           group="type",
           color="type"), data=losses) + \
    geom_line() + \
    labs(title="Convergence comparison") + \
    facet_wrap("activation") + \
    scale_x_continuous()