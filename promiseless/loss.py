import numpy


class LossFunction:
    @staticmethod
    def calculate(prediction: numpy.ndarray, target: numpy.ndarray):
        pass

    @staticmethod
    def derivative(prediction: numpy.ndarray, target: numpy.ndarray):
        pass


class MSE(LossFunction):
    @staticmethod
    def calculate(prediction: numpy.ndarray, target: numpy.ndarray):
        return numpy.sum((prediction - target) ** 2, axis=0) / prediction.shape[0]

    @staticmethod
    def derivative(prediction: numpy.ndarray, target: numpy.ndarray):
        return 2 * (prediction - target) / prediction.shape[0]


class MAE(LossFunction):
    @staticmethod
    def calculate(prediction: numpy.ndarray, target: numpy.ndarray):
        return numpy.sum(numpy.abs(prediction - target), axis=0) / prediction.shape[0]

    @staticmethod
    def derivative(prediction: numpy.ndarray, target: numpy.ndarray):
        return numpy.where(prediction - target >= 0, 1, -1) / prediction.shape[0]


class CategoricalCrossEntropy(LossFunction):
    @staticmethod
    def calculate(prediction: numpy.ndarray, target: numpy.ndarray):
        return -numpy.sum(target * numpy.log(1e-15 + prediction), axis=0) / prediction.shape[0]

    @staticmethod
    def derivative(prediction: numpy.ndarray, target: numpy.ndarray):
        return -numpy.sum(prediction / target, axis=0) / prediction.shape[0]
