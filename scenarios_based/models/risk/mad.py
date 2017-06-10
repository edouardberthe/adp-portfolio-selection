from gurobipy import quicksum
from time import time

import numpy as np
from matplotlib import pyplot as plt

from data import A, figsize
from scenarios_based.models.model import ScenariosBasedPortfolioModel


class MAD(ScenariosBasedPortfolioModel):
    """
    Implements Mean Absolute Deviation minimization model. Extends
    ScenariosBasedPortfolioModel, which itself extends gurobipy.Model
    """

    def __init__(self, scenarios, probas, name='MAD', *args, **kwargs):
        super().__init__(name, scenarios, probas, *args, **kwargs)

    def createVars(self):
        """Extends ScenariosBasedPortfolioModel.createVars and adds the variables specific to the MAD model."""
        super().createVars()
        self._D = [self.addVar(lb=0) for s in self._S]

    def createObjective(self):
        self.setObjective(
            quicksum(self._probas[s] * self._D[s] for s in self._S)
        )

    def createConstrs(self):
        super().createConstrs()
        self._cstr1 = [
            self.addConstr(self._D[s] >= self._Y[s] - self._mu)
            for s in self._S
        ]
        self._cstr2 = [
            self.addConstr(self._D[s] >= - (self._Y[s] - self._mu))
            for s in self._S
        ]

    def reconfigure(self, scenarios, probas):
        """
        This function is here to avoid re-creating the model several time, to save the time to add the variables.
        Updates the internal _scenarios and _probas via parent and updates the constraints coefficients.
        """
        # Calls ScenariosBasedPortfolioModel.reconfigure which reconfigures the RRR constraints, stores the new scenarios/probas
        # updates the internal _mu and _Y
        super().reconfigure(scenarios, probas)

        S = range(len(scenarios))
        MuA = probas.dot(scenarios)

        if self._output:
            t = time()
            print("Updating Constraints")
        for s in S:
            for a in A:
                self.chgCoeff(self._cstr1[s], self._W[a], - (scenarios[s, a] - MuA[a]))
                self.chgCoeff(self._cstr2[s], self._W[a], scenarios[s, a] - MuA[a])
        if self._output:
            print("\t{:.1f} s".format(time() - t))
            print("Updating objective")
        self.setObjective(
            quicksum(probas[s] * self._D[s] for s in S)
        )
        if self._output:
            print("\t{:.1f} s".format(time() - t))

        return self

    def plot(self):
        S, mu, Y = self._S, self._mu, self._Y

        # Plots the mean return of the output portfolio
        plt.figure(figsize=figsize)
        plt.plot(S, mu.getValue() * np.ones(len(S)))

        # Plots the absolute deviation from the mean
        for s in S:
            plt.plot([s], [Y[s].getValue()], 'o', color='red', label='Returns' if s == 0 else None)
            plt.plot([s, s], [Y[s].getValue(), mu.getValue()], color='red',
                     label='Absolute Mean Deviation' if s == 0 else None)
        plt.xlabel('Scenarios')
        plt.ylabel('Returns')
        plt.title('MAD Optimization with {:d} Scenarios'.format(len(S)))
        plt.legend()
        plt.show()
