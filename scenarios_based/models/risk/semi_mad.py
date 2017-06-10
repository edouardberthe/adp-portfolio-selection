from gurobipy import quicksum
from time import time

import numpy as np
from matplotlib import pyplot as plt

from data import A, figsize
from scenarios_based.models.model import ScenariosBasedPortfolioModel


class SemiMAD(ScenariosBasedPortfolioModel):
    """
    Implements Semi-Mean Absolute Deviation minimization model. Extends ScenariosBasedPortfolioModel, which itself extends
    gurobipy.Model.
    """

    def __init__(self, scenarios, probas, name='Semi-MAD', *args, **kwargs):
        super().__init__(name, scenarios, probas, *args, **kwargs)

    def createVars(self):
        """Adds the variables specific to the Semi-MAD model."""
        super().createVars()
        S = range(len(self._scenarios))
        self._D = [self.addVar(lb=0) for s in S]

    def createObjective(self):
        """Sets the objective value of the Semi-MAD Model."""
        self.setObjective(
            quicksum(self._probas[s] * self._D[s] for s in self._S)
        )

    def createConstrs(self):
        """Adds the constraints specific to the Semi-MAD Model."""
        super().createConstrs()
        self._cstr1 = [
            self.addConstr(
                self._D[s] >= self._Y[s] - self._mu
            ) for s in self._S
        ]

    def reconfigure(self, scenarios, probas):
        """
        This function is here to avoid re-creating the model several time, to save the time to add the variables.
        Updates the internal _scenarios and _probas, updates the internal LinExpr _Mu and _Y, and updates the
        constraints coefficients.
        """
        # Calls ScenariosBasedPortfolioModel.reconfigure which reconfigures the RRR constraints, stores the new scenarios/probas
        # updates the internal _mu and _Y
        super().reconfigure(scenarios, probas)

        S = range(len(scenarios))
        MuA = probas.dot(scenarios)

        if self._output:
            t = time()
            print("Updating Constr1")
        [self.chgCoeff(self._cstr1[s], self._W[a], - (scenarios[s, a] - MuA[a])) for a in A for s in S]
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
        plt.plot(S + [S[-1] + 1], mu.getValue() * np.ones(len(S) + 1))

        # Plots the negative deviations from the mean
        labelAdded = False
        for s in S:
            plt.plot([s], [Y[s].getValue()], 'o', color='red', label='Returns' if s == 0 else None)
            if Y[s].getValue() < 0:
                plt.plot([s, s], [Y[s].getValue(), mu.getValue()], color='red',
                         label='Negative Mean Deviation' if not labelAdded else None)
                labelAdded = True
        plt.xlabel('Scenarios')
        plt.ylabel('Returns')
        plt.title('Semi-MAD Optimization with {:d} scenarios'.format(len(S)))
        plt.legend()
        plt.show()
