from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import random as rd, ones

from adp.parameters import T, M
from data import N


class ValueFunction:

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, x) -> float:
        raise NotImplementedError


class LinearValueFunction(np.ndarray, ValueFunction):

    def __new__(cls, *args, **kwargs):
        return rd.rand(T+1, N+1).view(cls)

    def __call__(self, x):
        return (self * x).sum()


class PWLinearValueFunction(ValueFunction):

    def __init__(self):
        self.slopes = np.ones((T, N+1, M+1))
        self.cash = 1.

    def __call__(self, x) -> float:
        return x[0] * self.cash + x[1:].dot(self.slopes)
