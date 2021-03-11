from numpy.random import random


class RandomInitialization:
    def __init__(self):
        pass

    @staticmethod
    def perform(shape):
        return random(shape)
