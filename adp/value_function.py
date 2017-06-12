from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import random as rd

from data import N, Data
from parameters import M, T, a, decimals, m, w0


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


class PWLFunction:

    __metaclass__ = ABCMeta

    def __call__(self, h: float) -> float:
        pi = self.pi(h)
        return self.y()[pi] + (h - self.a[pi]) * self.slopes[pi]

    def __str__(self):
        return "PWL Function"

    def pi(self, h: float) -> int:
        try:
            return np.where(self.a <= h)[0][-1]
        except IndexError:
            pass

    def x(self):
        return np.concatenate((self.a, (self.a[-1]+10,)))

    def y(self):
        x = self.x()
        return np.concatenate((
            (0,),
            np.cumsum(self.slopes * (x[1:] - x[:-1]))
        ))


class PWLDynamicFunction(PWLFunction):

    def __init__(self):
        self.slopes = np.ones(1)
        self.a = np.zeros(1)

    def __str__(self):
        return "{:s} (Dynamic)".format(super().__str__())

    def update(self, h: float, deltaV: float, alpha: float):

        h = round(h, decimals)

        pi = self.pi(h)

        new_slope = (1 - alpha) * self.slopes[pi] + alpha * deltaV

        # Checking h and a
        if h in self.a:
            self.slopes[pi] = new_slope
        else:
            # If the state has never been reached, we add a cut
            pi += 1
            self.a = np.insert(self.a, pi, h)
            self.slopes = np.insert(self.slopes, pi, new_slope)

        # Checking if slopes are still decreasing
        # First Case: monotonicity failed on the left
        if pi > 0 and self.slopes[pi - 1] < self.slopes[pi]:
            k = pi - 1
            while True:
                # We select from k to pi (we have to put +1)
                updated = self.slopes[k:pi + 1].mean()
                if k == 0 or updated < self.slopes[k - 1]:
                    self.slopes = np.delete(self.slopes, slice(k + 1, pi + 1))
                    self.a = np.delete(self.a, slice(k + 1, pi + 1))
                    self.slopes[k] = updated
                    break
                else:
                    k -= 1

        # Second Case: monotonicity failed on the left
        elif pi < len(self.slopes) - 1 and self.slopes[pi] < self.slopes[pi + 1]:
            k = pi + 1
            while True:
                updated = self.slopes[pi:k + 1].mean()
                if k == len(self.slopes) - 1 or updated >= self.slopes[k + 1]:
                    self.slopes = np.delete(self.slopes, slice(pi + 1, k + 1))
                    self.a = np.delete(self.a, slice(pi + 1, k + 1))
                    self.slopes[pi] = updated
                    break
                else:
                    k += 1

        # Check the number of slopes
        if m is not None and len(self.slopes) > m:
            k = (self.slopes[:-1] - self.slopes[1:]).argmin()
            self.slopes[k] = self.slopes[k:k+2].mean()
            self.slopes = np.delete(self.slopes, k+1)
            self.a = np.delete(self.a, k+1)


class PWLFixedFunction(PWLFunction):

    def __init__(self):
        self.slopes = np.ones(M)
        self.a = a

    def __str__(self):
        return "{:s} (Dynamic)".format(super().__str__())

    def update(self, h: float, deltaV: float, alpha: float):

        pi = self.pi(h)

        self.slopes[pi] = (1 - alpha) * self.slopes[pi] + alpha * deltaV

        # Checking if slopes are still decreasing
        # First Case: monotonicity failed on the left
        if pi > 0 and self.slopes[pi-1] < self.slopes[pi]:
            k = pi-1
            while True:
                # We select from k to pi (we have to put +1)
                updated = self.slopes[k:pi+1].mean()
                if k == 0 or updated < self.slopes[k-1]:
                    self.slopes[k:pi+1] = updated
                    break
                else:
                    k -= 1

        # Second Case: monotonicity failed on the left
        elif pi < len(self.slopes)-1 and self.slopes[pi] < self.slopes[pi+1]:
            k = pi+1
            while True:
                updated = self.slopes[pi:k+1].mean()
                if k == len(self.slopes)-1 or updated >= self.slopes[k+1]:
                    self.slopes[pi:k+1] = updated
                    break
                else:
                    k += 1


class SeparableValueFunction(ValueFunction):

    def __init__(self, value_function_class=PWLDynamicFunction):
        super().__init__()
        self.value_functions = pd.Series([value_function_class() for _ in range(N)], index=Data.columns)
        self.cash = 1.

    def __call__(self, h: np.ndarray) -> np.ndarray:
        return np.array([h[0] * self.cash] + [self[i](x) for (i, x) in enumerate(h[1:])])

    def __str__(self):
        return "Separable Value Function ({:d} {:s})".format(len(self.value_functions),
                                                             self[0].__str__())

    def __getitem__(self, item):
        return self.value_functions[item]

    def __iter__(self):
        return iter(self.value_functions)

    def update(self, h, deltaV, alpha):
        # Update the cash slope
        self.cash = (1 - alpha) * self.cash + alpha * deltaV[0]

        # Updating the assets positions' value functions
        for i in range(N):
            self[i].update(h[i+1], deltaV[i+1], alpha)


if __name__ == '__main__':
    V = SeparableValueFunction()
    plt.ion()
    for s in range(100):
        h = w0 * rd.rand(N+1)
        deltaV = rd.rand(N+1)
        alpha = 0.5
        V.update(h, deltaV, alpha)
        plt.clf()
        for i in range(N):
            plt.plot(V[i].a, V[i].y())
        plt.pause(0.5)
