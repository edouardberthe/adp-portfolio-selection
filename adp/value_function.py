from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import random as rd, ones

from adp.parameters import M, T, init
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
        # self.slopes = 1 / rd.rand(M, N).cumsum(axis=0)
        self.slopes = ones((M, N))
        self.cash = 1.
        self.a = np.tile(np.linspace(0, init, M), (N, 1)).T

    def __call__(self, h: np.ndarray) -> float:
        x = h[1:]
        idx = self.idx(x)[1:]                        # Warning: size M-1
        k = np.argmin(self.idx(x), axis=0) - 1
        diff_u = self.slopes[:-1] - self.slopes[1:]  # Same
        d = (diff_u[idx] * self.a[1:][idx]).sum()    # Sum of all intercepts size N
        u = self.slopes[k, range(N)]
        return h[0] * self.cash + d + x.dot(u)

    def idx(self, x):
        """Warning: h should be of size N (not N+1): only assets, no cash."""
        x_mat = np.tile(x, (M, 1))
        return self.a <= x_mat

    def update(self, ΔV, pi, alpha):
        self.cash = (1 - alpha) * self.cash + alpha * ΔV[0]
        self.slopes[pi, range(N)] = (1 - alpha) * self.slopes[pi, range(N)] + alpha * ΔV[1:]
