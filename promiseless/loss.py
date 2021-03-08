import numpy as np


class LossFunction:
    def __init__(self):
        pass

    @staticmethod
    def calculate(prediction, target):
        pass

    @staticmethod
    def derivative(prediction, target):
        pass


class MSE(LossFunction):
    def __init__(self):
        super().__init__()

    @staticmethod
    def calculate(prediction, target):
        return np.sum((prediction - target)**2, axis=0)/prediction.shape[0]

    @staticmethod
    def derivative(prediction, target):
        return 2*(prediction - target)/prediction.shape[0]