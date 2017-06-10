from itertools import count

import numpy as np
from adp.updater import generateΔCVaR
from numpy import array, ones, zeros

from adp.generator import GaussianGenerator
from data import N
from parameters import T, alpha, gamma, init


class LValueFunctionUpdater:

    def __init__(self, V, m):
        self.V = V
        self.m = m
        self.generator = GaussianGenerator()
        self.h_plus = array([], dtype=np.int64).reshape(0, N+1)
        self.RT = array([], dtype=np.int64).reshape(0, N+1)
        self.counter = count()

    def __next__(self):
        s = next(self.counter)
        print("Scenario", s)

        # Initialization
        deltaV = zeros((T, N+1))
        h_plus = zeros(N+1)
        h_plus[0] = init

        print("\tTime 0")
        self.m.solve(ones(N+1), h_plus, self.V[0])
        h_plus = self.m.h_plus

        # 1 <= t <= T - 1
        for t in range(1, T):
            self.m.solve(self.generator.generate(), h_plus, self.V[t])
            h_plus = self.m.h_plus
            deltaV[t-1] = self.m.deltaV

        # Last returns (which we store for later scenarios)
        self.RT = np.vstack((self.RT, self.generator.generate()))
        self.h_plus = np.vstack((self.h_plus, h_plus))

        # t = T
        print("\tLast time: t =", T)
        ΔCVaR = generateΔCVaR(self.RT[:s+1], self.h_plus[:s+1])
        deltaV[T-1] = (1 - gamma) * self.RT[s] - gamma * ΔCVaR

        self.V[:] = (1 - alpha[s]) * self.V + alpha[s] * deltaV
