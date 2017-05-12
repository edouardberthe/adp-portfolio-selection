from abc import ABCMeta, abstractmethod
from itertools import count

from numpy import identity, ones, zeros

from adp.cvar import CVaR
from adp.generator import GaussianGenerator
from adp.parameters import S, T, alpha, gamma, init, M
from data import N

e = identity(N+1)


def generateΔCVaR(RT, h_plus):
    CVaR_plus = zeros(N+1)
    for i in range(N+1):
        CVaR_plus[i] = CVaR(RT * (h_plus + [e[i]]))
    return CVaR_plus - CVaR(RT * h_plus)


class ValueFunctionUpdater(object):

    __metadata__ = ABCMeta

    def __init__(self, V, m):
        self.V = V
        self.m = m
        self.generator = GaussianGenerator()
        self.h_plus = zeros((S, N+1))
        self.h_plus[:, 0] = init
        self.RT = zeros((S, N+1))
        self.counter = count()

    @abstractmethod
    def __next__(self):
        raise NotImplementedError


class LValueFunctionUpdater(ValueFunctionUpdater):

    def __next__(self):
        s = next(self.counter)
        print("Scenario", s)

        ΔV = zeros((T, N+1))

        print("\tTime 0")
        self.m.solve(ones(N+1), self.h_plus[s], self.V[0])
        self.h_plus[s] = self.m.h_plus

        # 1 <= t <= T - 1
        for t in range(1, T):
            R = self.generator.generate()
            self.m.solve(R, self.h_plus[s], self.V[t])
            self.h_plus[s] = self.m.h_plus
            ΔV[t-1] = self.m.ΔV

        # Last returns (which we store for later scenarios)
        self.RT[s] = self.generator.generate()

        # t = T
        print("\tLast time: t =", T)
        ΔCVaR = generateΔCVaR(self.RT[:s+1], self.h_plus[:s+1])
        ΔV[T-1] = (1 - gamma) * self.RT[s] - gamma * ΔCVaR

        self.V[:] = (1 - alpha[s]) * self.V + alpha[s] * ΔV


class PWLValueFunctionUpdater(ValueFunctionUpdater):

    def __next__(self):
        s = next(self.counter)
        print("Scenario", s)

        print("\tTime 0")
        self.m.solve(ones(N + 1), self.h_plus[s], self.V[0])
        self.h_plus[s] = self.m.h_plus

        # 1 <= t <= T - 1
        for t in range(1, T):
            R = self.generator.generate()
            self.m.solve(R, self.h_plus[s], self.V[t])
            self.h_plus[s] = self.m.h_plus
            self.V[t-1].update(self.m.ΔV, self.m.pi, alpha[s])

        # Last returns (which we store for later scenarios)
        self.RT[s] = self.generator.generate()

        # t = T
        print("\tLast time: t =", T)
        ΔCVaR = generateΔCVaR(self.RT[:s + 1], self.h_plus[:s + 1])
        ΔV = (1 - gamma) * self.RT[s] - gamma * ΔCVaR
        self.V[T-1].update(ΔV, zeros(N)==0, alpha[s])

