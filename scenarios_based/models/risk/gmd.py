from gurobipy import quicksum
from time import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from data import A, figsize
from scenarios_based.models.model import ScenariosBasedPortfolioModel
from generator import generateGaussianScenarios


class GMD(ScenariosBasedPortfolioModel):
    """
    Implements Gini's Mean Deviation minimization model. Extends ScenariosBasedPortfolioModel, which itself extends
    gurobipy.Model.
    """
    def __init__(self, scenarios, probas, name='GMD', *args, **kwargs):
        super().__init__(name, scenarios, probas, *args, **kwargs)

    def createVars(self):
        """Adds the variables specific to the GMD model."""
        super().createVars()
        self._D = {
            (s1, s2): self.addVar(lb=0)
            for s1 in self._S for s2 in self._S if s1 < s2
        }

    def createObjective(self):
        self.setObjective(
            2 * quicksum(
                self._probas[s1] * self._probas[s2] * self._D[s1, s2] for s1, s2 in self._D
            )
        )

    def createConstrs(self):
        super().createConstrs()
        self._cstr = {
            (s1, s2): self.addConstr(self._D[s1, s2] >= self._Y[s1] - self._Y[s2])
            for s1, s2 in self._D

        }

    def reconfigure(self, scenarios, probas):
        """
        This function is here to avoid re-creating the model several time, to save the time to add the variables.
        Updates the internal _scenarios and _probas, updates the internal LinExpr _Mu and _Y, and updates the
        constraints coefficients.
        """
        # Calls ScenariosBasedPortfolioModel.reconfigure which reconfigures the RRR constraints, stores the new scenarios/probas
        # updates the internal _mu and _Y
        super().reconfigure(scenarios, probas)

        if self._output:
            t = time()
            print("Updating Constr")
        for a in A:
            for s1, s2 in self._D:
                self.chgCoeff(self._cstr[s1, s2], self._W[a], - (scenarios[s1, a] - scenarios[s2, a]))
        if self._output:
            print("\t{:.2f} s".format(time() - t))
            print("Updating objective")
        self.setObjective(
            2 * quicksum(
                probas[s1] * probas[s2] * self._D[s1, s2] for s1, s2 in self._D
            )
        )
        if self._output:
            print("\t{:.2f} s".format(time() - t))

        return self

    def plot(self):
        S, mu, Y = self._S, self._mu, self._Y

        # Plots the mean return of the output portfolio
        plt.figure(figsize=figsize)
        plt.plot(S, mu.getValue() * np.ones(len(S)))

        # Plots the absolute deviation from the mean

        for s in S:
            plt.plot([s], [Y[s].getValue()], 'o', color='red', label='Returns' if s == 0 else None)

        values = np.array([Y[s].getValue() for s in S])
        i = int(len(values)/2)
        for value in [values[:i], values[i:]]:
            Min = np.argwhere(values == np.percentile(value, 1, interpolation='higher'))[0, 0]
            Max = np.argwhere(values == np.percentile(value, 99, interpolation='lower'))[0, 0]
            middle = (Min + Max) / 2
            plt.annotate('', xy=(middle, values[Min]), xytext=(middle, values[Max]), fontsize=7, color='red',
                         arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0, lw=2, color='red'))

            plt.plot([Min, middle], [values[Min], values[Min]], '--', color='red', lw=2)
            plt.plot([middle, Max], [values[Max], values[Max]], '--', color='red', lw=2)

        plt.xlabel('Scenarios')
        plt.ylabel('Returns')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        plt.title('GMD Optimization with {:d} scenarios'.format(len(S)))
        plt.legend()
        plt.show()

if __name__ == '__main__':
    GMD(*generateGaussianScenarios(100)).optimize().plot()
