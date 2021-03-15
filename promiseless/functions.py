import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def visualize_results(x_test, result, y_test, task):
    if task == "regression":
        plt.figure()
        plt.plot(x_test, result, label="Fitted values")
        plt.plot(x_test, y_test, label="Original values")
        plt.title("Fitted vs original")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
    elif task == "classification":
        plt.figure()
        plt.subplot(121)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=result)
        plt.title("Fitted test set")

        plt.subplot(122)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=np.argmax(y_test, axis=1))
        plt.title("Original test set")
        plt.show()


def visualize_loss(model):
    plt.figure()
    plt.plot(range(1, len(model.training_history[0])+1), model.training_history[0], label="Training set")
    if len(model.training_history[1]):
        plt.plot(range(1, len(model.training_history[1])+1), model.training_history[1], label="Evaluation set")
    plt.title("Loss during training")
    plt.xlabel("Number of epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()