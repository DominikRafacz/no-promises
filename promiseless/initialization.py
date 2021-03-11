from numpy.random import random


class InitializationMethod:
    @staticmethod
    def perform(shape):
        pass


class RandomInitialization(InitializationMethod):
    @staticmethod
    def perform(shape):
        return random(shape)
