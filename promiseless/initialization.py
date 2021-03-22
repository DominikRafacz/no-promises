from numpy.random import random, uniform
import numpy as np


class InitializationMethod:
    @staticmethod
    def perform(shape):
        pass


class RandomInitialization(InitializationMethod):
    @staticmethod
    def perform(shape):
        return random(shape)


class XavierInitialization(InitializationMethod):
    @staticmethod
    def perform(shape):
        limit = np.sqrt(6) / np.sqrt(np.sum(shape))
        return uniform(low=-limit, high=limit, size=shape)