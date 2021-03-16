import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.ioff()

def read_data(task, dataset_name, path_to_data=""):
    data = pd.read_csv("{0}data/{1}/{2}.csv".format(path_to_data, task, dataset_name))
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


def visualize_results(x_test, result, y_test, task, filename=None):
    fig = plt.figure()
    if task == "regression":
        plt.plot(x_test, result, label="Fitted values")
        plt.plot(x_test, y_test, label="Original values")
        plt.title("Fitted vs original")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
    elif task == "classification":
        plt.subplot(121)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=np.argmax(y_test, axis=1))
        plt.title("Original test set")

        plt.subplot(122)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=result)
        plt.title("Fitted test set")
    if filename:
        plt.savefig("plots/{}_{}".format(filename, task[:4]))
        plt.close(fig)
    else:
        plt.show()


def visualize_loss(model, filename=None):
    fig = plt.figure()
    plt.plot(range(1, len(model.training_history[0])+1), model.training_history[0], label="Training set")
    if len(model.training_history[1]):
        plt.plot(range(1, len(model.training_history[1])+1), model.training_history[1], label="Evaluation set")
    plt.title("Loss during training")
    plt.xlabel("Number of epoch")
    plt.ylabel("Loss")
    plt.legend()
    if filename:
        plt.savefig("plots/{}_loss".format(filename))
        plt.close(fig)
    else:
        plt.show()


def visualize_losses(models, labels=None, data="train", start_from=0, filename=None, styles=None):
    fig = plt.figure()
    if not labels:
        labels = ["Model {0}".format(j) for j in range(1, len(models)+1)]
    if styles is None:
        for model, label in zip(models, labels):
            i = 0 if data == "train" else 1
            plt.plot(range(start_from+1, len(model.training_history[i])+1), model.training_history[i][start_from:], label="{0}".format(label))
            plt.title("Loss during training")
            plt.xlabel("Number of epoch")
            plt.ylabel("Loss")
            plt.legend()
    else:
        for model, label, style in zip(models, labels, styles):
            i = 0 if data == "train" else 1
            plt.plot(range(start_from + 1, len(model.training_history[i]) + 1), model.training_history[i][start_from:],
                     label="{0}".format(label), color=style['color'], linestyle=style['linestyle'])
            plt.title("Loss during training")
            plt.xlabel("Number of epoch")
            plt.ylabel("Loss")
            plt.legend()
    if filename:
        plt.savefig("plots/{}_loss".format(filename))
        plt.close(fig)
    else:
        plt.show()


def visualize_results2(x_test, results, y_test, task, labels=None, filename=None):
    fig = plt.figure()
    n = len(results)
    if not labels:
        labels = ["Model {0}".format(j) for j in range(1, n+1)]
    if task == "regression":
        for result, label in zip(results, labels):
            plt.plot(x_test, result, label="{}".format(label))
        plt.plot(x_test, y_test, label="Original values", linewidth=4.0)
        plt.title("Fitted vs original")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
    elif task == "classification":
        m = n//3 + 1
        plt.subplot(m,3,1)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=np.argmax(y_test, axis=1), s=3)
        plt.title("Original test set")
        for i in range(2, n+2):
            plt.subplot(m,3,i)
            plt.scatter(x_test[:, 0], x_test[:, 1], c=results[i-2], s=3)
            plt.title("{0}".format(labels[i-2]))
    if filename:
        plt.savefig("plots/{}_{}".format(filename, task[:4]))
        plt.close(fig)
    else:
        plt.show()