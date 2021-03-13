import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from promiseless.Architecture import Architecture
from promiseless.layer import InputLayer, HiddenLayer
from promiseless.activation import Sigmoid, ReLU
from promiseless.loss import CategoricalCrossEntropy

def read_data(task, dataset_name):
    data = pd.read_csv("data/{0}/{1}.csv".format(task, dataset_name))
    if task == "regression":
        x_train = data.loc[:, ["x"]]
        x_train = np.array(x_train)
        x_train = (x_train - x_train.mean()) / x_train.std()
        y_train = np.array(data.loc[:, ["y"]])
        y_train = (y_train - y_train.mean()) / y_train.std()
    elif task == "classification":
        x_train = data.loc[:, ["x", "y"]]
        x_train = np.array(x_train)
        x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
        y_train = np.array(pd.get_dummies(data.loc[:, "cls"]))
    else:
        print("Unknown task")
        x_train = data
        y_train = None
    return x_train, y_train


name = "data.cube.train.100"
x_train, y_train = read_data("regression", name)
x_test, y_test = read_data("regression", "data.cube.test.100")


name2 = "data.simple.train.100"
x_train2, y_train2 = read_data("classification", name2)

np.random.seed(123)

mdl = Architecture()\
    .add_input_layer(InputLayer(1))\
    .add_layer(HiddenLayer(5, activation=Sigmoid))\
    .add_layer(HiddenLayer(1))\
    .build_model()

mdl.train(x_train, y_train, batch_size=100, learning_rate=10e-4, epochs=100, evaluation_dataset=(x_test, y_test))

res, loss = mdl.predict(x_test, y_test)

np.random.seed(123)

mdl_classif = Architecture()\
    .add_input_layer(InputLayer(2))\
    .add_layer(HiddenLayer(5, activation=Sigmoid))\
    .add_layer(HiddenLayer(2, activation=Sigmoid))\
    .set_loss_function(CategoricalCrossEntropy)\
    .build_model()

mdl_classif.train(x_train2, y_train2, batch_size=100, learning_rate=10e-4, epochs=100)

res2, loss2 = mdl_classif.predict(x_train2, y_train2, return_class=True)


def visualize_results(x_test, result, y_test, task):
    if task == "regression":
        fig, ax = plt.subplots()
        ax.plot(x_test, result, label="Fitted values")
        ax.plot(x_test, y_test, label="Original values")
        legend = ax.legend()
        ax.set_title("Fitted test set")
        # ax.xlabel("X")
        # ax.ylabel("Y")
        plt.show()
    elif task == "classification":
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.scatter(x_test[:,0], x_test[:,1], c=result)
        ax1.set_title("Fitted test set")
        # ax1.xlabel("X")
        # ax1.ylabel("Y")
        ax2.scatter(x_test[:,0], x_test[:,1], c=y_test-1)
        ax2.set_title("Original test set")
        # ax2.xlabel("X")
        # ax2.ylabel("Y")
        plt.show()


visualize_results(x_test, res, y_test, "regression")
visualize_results(x_train2, res2, y_train2, "classification")

plt.scatter(x_test, res)
plt.show()
print(mdl.training_history)

plt.plot(range(1,101),mdl.training_history[0])
plt.plot(range(1,101),mdl.training_history[1])