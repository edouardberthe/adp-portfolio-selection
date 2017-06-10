from gurobipy import GRB, quicksum
from time import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from data import figsize, A
from scenarios_based.models.model import ScenariosBasedPortfolioModel
from generator import generateGaussianScenarios


class CVaR(ScenariosBasedPortfolioModel):
    """
    Implements 'Conditional Value-at-Risk' model (CVaR), also called 'Tail Mean' or 'Worst conditional expectation'.
    This model aims at maximizing the mean return of the beta % worst returns.
    Extends ScenariosBasedPortfolioModel, which itself extends gurobipy.Model.
    """

    def __init__(self, scenarios, probas, beta=0.1, name='CVaR', *args, **kwargs):
        self._beta = beta
        super().__init__(name, scenarios, probas, *args, **kwargs)

    def createVars(self):
        """Adds the variables specific to the CVaR model."""
        super().createVars()
        self._eta = self.addVar(lb=-GRB.INFINITY)
        self._d = [self.addVar(lb=0) for s in self._S]

    def createObjective(self):
        self.setObjective(
            self._eta - quicksum(self._probas[s] * self._d[s] for s in self._S) / self._beta,
            GRB.MAXIMIZE
        )

    def createConstrs(self):
        super().createConstrs()
        self._cstr = [
            self.addConstr(
                self._d[s] >= self._eta - self._Y[s]
            ) for s in self._S
        ]

    def reconfigure(self, scenarios, probas):
        super().reconfigure(scenarios, probas)
        S = range(len(scenarios))

        if self._output:
            t = time()
            print("Updating Constr")
        for s in S:
            for a in A:
                self.chgCoeff(self._cstr[s], self._W[a], scenarios[s, a])
        if self._output:
            print("\t{:.1f} s".format(time() - t))
            print("Updating objective")
        if self._output:
            print("\t{:.1f} s".format(time() - t))
        self.createObjective()
        return self

    def plot(self):
        S, mu, Y = np.array(self._S), self._mu, self._Y

        # Plots the CVaR
        returns = np.array([Y[s].getValue() for s in S])
        quantile = np.percentile(returns, self._beta * 100)
        worstIndex = S[returns <= quantile]
        bestIndex = S[returns > quantile]
        mean = returns[worstIndex].mean()

        # We compute the case of an equally-weighted portfolio
        equally = self._scenarios.mean(axis=1)
        oldQuantile = np.percentile(equally, self._beta * 10)
        oldWorstIndex = S[equally <= oldQuantile]
        oldMean = equally[oldWorstIndex].mean()

        # Plots the bets returns
        plt.figure(figsize=figsize)
        plt.plot(bestIndex, returns[bestIndex], 'o', color='green', label='Best Cases', alpha=0.5)

        # Plots the beta % worst returns
        plt.plot(worstIndex, returns[worstIndex], 'o', color='red', label='{:.0%} Worst Cases'.format(self._beta))
        plt.plot(oldWorstIndex, equally[oldWorstIndex], 'o', color='orange', label='E.W. {:.0%} Worst Cases'.format(self._beta))

        # Plots the mean of the beta % worst returns in equally weighted portfolio
        plt.plot([S[0], S[-1]], [oldMean, oldMean], '-', color='orange')
        # Plots the mean of the beta % worst returns
        plt.plot([S[0], S[-1]], [mean, mean], '-', color='red')

        plt.xlabel('Scenarios')
        plt.ylabel('Returns')
        plt.title('Conditional Value-at-Risk Optimization with {:d} scenarios'.format(len(S)))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        plt.legend()
        plt.show()

if __name__ == '__main__':
    s, p = generateGaussianScenarios(300)
    CVaR(s, p, output=True).optimize().plot()
