import numpy


class ActivationFunction:
    @staticmethod
    def calculate(data: numpy.ndarray):
        pass

    @staticmethod
    def derivative(data: numpy.ndarray):
        pass


class LinearActivation(ActivationFunction):
    @staticmethod
    def calculate(data: numpy.ndarray):
        return data

    @staticmethod
    def derivative(data: numpy.ndarray):
        return numpy.ones(data.shape)


class Sigmoid(ActivationFunction):
    @staticmethod
    def calculate(data: numpy.ndarray):
        return 1 / (1 + numpy.exp(-data))

    @staticmethod
    def derivative(data: numpy.ndarray):
        return Sigmoid.calculate(data) * (1 - Sigmoid.calculate(data))


class ReLU(ActivationFunction):
    @staticmethod
    def calculate(data: numpy.ndarray):
        return numpy.where(data < 0, 0, data)

    @staticmethod
    def derivative(data: numpy.ndarray):
        return numpy.where(data < 0, 0, 1)


class Tanh(ActivationFunction):
    @staticmethod
    def calculate(data: numpy.ndarray):
        return numpy.tanh(data)

    @staticmethod
    def derivative(data: numpy.ndarray):
        return 1 - numpy.tanh(data) ** 2


class Softmax(ActivationFunction):
    @staticmethod
    def calculate(data: numpy.ndarray):
        data = numpy.exp(data)
        return data / numpy.sum(data, axis=1)

    @staticmethod
    def derivative(data: numpy.ndarray):
        return Softmax.calculate(data) * (1 - Softmax.calculate(data))
