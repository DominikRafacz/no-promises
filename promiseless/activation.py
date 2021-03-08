import numpy as np


class LinearActivation:
    def __init__(self):
        pass

    @staticmethod
    def calculate(data):
        return data

    @staticmethod
    def derivative(data):
        return np.ones(data.shape)


class Sigmoid:
    def __init__(self):
        pass

    @staticmethod
    def calculate(data):
        return 1 / (1 + np.exp(-data))

    @staticmethod
    def derivative(data):
        return Sigmoid.calculate(data)*(1-Sigmoid.calculate(data))