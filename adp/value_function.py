from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import random as rd, ones, zeros

from adp.parameters import M, T, init, theta
from data import N


class ValueFunction:

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, h: np.ndarray) -> float:
        """Returns the value attribued on the portfolio h"""
        raise NotImplementedError


class LinearValueFunction(np.ndarray, ValueFunction):

    def __new__(cls, *args, **kwargs):
        return rd.rand(T, N+1).view(cls)

    def __call__(self, h: np.ndarray) -> float:
        return (self * h).sum()


class PWLinearValueFunction(ValueFunction):

    def __init__(self):
        self.slopes = 1 / rd.rand(M, N).cumsum(axis=0)
        # self.slopes = ones((M, N))
        self.cash = 0.001
        self.a = np.tile(np.linspace(0, init, M), (N, 1)).T

    def __call__(self, h: np.ndarray) -> float:
        x = h[1:]
        pi = self.pi(x)
        return h[0] * self.cash \
            + self.y()[pi, range(N)].sum() \
            + ((x - self.a[pi, range(N)]) * self.slopes[pi, range(N)]).sum()

    def idx(self, x):
        """Warning: h should be of size N (not N+1): only assets, no cash."""
        x_mat = np.tile(x, (M, 1))
        return self.a <= x_mat

    def pi(self, x):
        return np.argmin(self.idx(x), axis=0) - 1

    def update(self, ΔV, pi, alpha):
        self.cash = (1 - alpha) * self.cash + alpha * ΔV[0]
        self.slopes[pi, range(N)] = (1 - alpha) * self.slopes[pi, range(N)] + alpha * ΔV[1:]

    def buying_y(self):
        return np.vstack((
            zeros(N),
            (self.slopes[:-1] - (1+theta) * self.cash) * (self.a[1:] - self.a[:-1])
        )).cumsum(axis=0)

    def selling_y(self):
        return np.vstack((
            zeros(N),
            (- self.slopes[:-1] + (1-theta) * self.cash) * (self.a[1:] - self.a[:-1])
        )).cumsum(axis=0)

    def y(self):
        return np.vstack((
            zeros(N),
            self.slopes[:-1] * (self.a[1:] - self.a[:-1])
        )).cumsum(axis=0)
