from itertools import count

from numpy import identity, ones
from numpy.ma import zeros

from adp.cvar import CVaR
from adp.parameters import S, T, alpha, gamma, init
from data import N

e = identity(N+1)


def generateΔCVaR(RT, h_plus):
    CVaR_hat = zeros(N+1)
    for i in range(N+1):
        CVaR_hat[i] = CVaR(RT * (h_plus + [e[i]]))
    return CVaR_hat - CVaR(RT * h_plus)


class ValueFunctionUpdater(object):

    def __init__(self, V, m, generator):
        self.V = V
        self.m = m
        self.generator = generator
        self.h_plus = zeros((S, N+1))
        self.h_plus[:, 0] = init
        self.RT = zeros((S, N+1))
        self.counter = count()

    def __next__(self):
        s = next(self.counter)
        print("Scenario", s)

        ΔV = zeros((T, N+1))

        print("\tTime 0")
        self.m.solve(ones(N+1), self.h_plus[s], self.V[0])
        self.h_plus[s] = self.m.h_plus
        # ΔV[0] = self.m.ΔV

        # 1 <= t <= T - 1
        for t in range(1, T):
            self.m.solve(self.generator.generate(), self.h_plus[s], self.V[t])
            self.h_plus[s] = self.m.h_plus
            ΔV[t-1] = self.m.ΔV

        # Last returns (which we store for later scenarios)
        self.RT[s] = self.generator.generate()

        # t = T
        print("\tLast time: t =", T)
        ΔCVaR = generateΔCVaR(self.RT[:s+1], self.h_plus[:s+1])
        ΔV[T-1] = gamma * self.RT[s] - (1 - gamma) * ΔCVaR

        hT = self.RT[:s+1] * self.h_plus[:s+1]
        self.V[T] = gamma * hT[s] - (1 - gamma) * CVaR(hT)
        self.V[:T] = (1 - alpha[s]) * self.V[:T] + alpha[s] * ΔV
