from gurobipy import GurobiError

import numpy as np
from numpy import array, ones, zeros

from adp.cvar import generateΔCVaR
from adp.generator import Generator
from adp.pwladp.model import gurobiModel
from data import N
from parameters import T, alpha, init


class TrainingMemory:
    def __init__(self):
        self.hp = array([], dtype=np.float64).reshape(0, N+1)
        self.RT = array([], dtype=np.float64).reshape(0, N+1)
        self.h = array([], dtype=np.float64)


class ADPStrategyTrainer:

    def __init__(self, gamma: float, generator: Generator):
        self.gamma = gamma
        self.generator = generator

        self.counter = 0
        self.memory = TrainingMemory()

    def train(self, strategy):
        alpha_s = alpha(self.counter)

        # Initialization
        hp = zeros(N+1)
        hp[0] = init
        R = self.generator.generate(T)

        # t = 0
        hp, ΔV = gurobiModel(ones(N+1), hp, strategy[0])

        # 1 <= t <= T - 1
        for t in range(1, T):
            old = hp
            try:
                hp, ΔV = gurobiModel(R[t-1], hp, strategy[t])
            except GurobiError as e:
                print(e.message)
                hp = R * hp
            else:
                strategy[t-1].update(old, ΔV, alpha_s)

        # t = T
        h = R[T-1] * hp  # Final Wealth

        # Updating training memory
        self.memory.RT = np.vstack((self.memory.RT, R[T-1]))
        self.memory.hp = np.vstack((self.memory.hp, hp))
        self.memory.h = np.append(self.memory.h, h.sum())

        ΔCVaR = generateΔCVaR(self.memory.RT, self.memory.hp)
        ΔV = self.gamma * R[T-1] - (1 - self.gamma) * ΔCVaR
        strategy[T-1].update(h, ΔV, alpha_s)
        self.counter += 1
