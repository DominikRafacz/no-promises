from numpy.random import normal


class RandomInitialization:
    def __init__(self):
        pass

    @staticmethod
    def perform(shape):
        return normal(size=shape)
