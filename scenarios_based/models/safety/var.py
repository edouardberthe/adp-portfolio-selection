from gurobipy import GRB, quicksum
from time import time

from matplotlib import pyplot as plt

from data import A
from scenarios_based.models.model import ScenariosBasedPortfolioModel


class VaR(ScenariosBasedPortfolioModel):
    """
    Implements VaR model. Extends ScenariosBasedPortfolioModel, which itself extends gurobipy.Model.
    """

    def __init__(self, scenarios, probas, beta=0.1, name='VaR', *args, **kwargs):
        self._beta = beta
        super().__init__(name, scenarios, probas, *args, **kwargs)

    def createVars(self):
        """Extends parent and adds the variables specific to the VaR model."""
        super().createVars()
        self._My = self.addVar(lb=-1)        # a return of - 100% is really unlikely
        self._Z = [self.addVar(vtype=GRB.BINARY) for s in self._S]

    def createObjective(self):
        self.setObjective(self._My, GRB.MAXIMIZE)

    def createConstrs(self):
        super().createConstrs()
        eps = self._probas.min() / 10
        M = 1
        self._cstr1 = [
            self.addConstr(
                self._Y[s] >= self._My - M * self._Z[s]
            ) for s in self._S
        ]
        self._cstr2 = self.addConstr(
            quicksum(self._probas[s] * self._Z[s] for s in self._S) <= self._beta - eps
        )

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
            print("Updating Constraints 1")
        for a in A:
            for s in self._S:
                self.chgCoeff(self._cstr1[s], self._W[a], scenarios[s, a])
        if self._output:
            print("\t{:.2f} s".format(time() - t))
            t = time()
            print("Updating Constraint 2")
        for s in self._S:
            self.chgCoeff(self._cstr2, self._Z[s], probas[s])
        eps = probas.min() / 10
        self._cstr2.rhs = self._beta - eps
        if self._output:
            print("\t{:.2f} s".format(time() - t))

        return self

    def plot(self):
        S, mu, Y = self._S, self._mu, self._Y

        # Plots the VaR
        minis = [s for s in S if abs(Y[s].getValue() - self.objVal) < 10e-5]
        plt.plot(S, [Y[s].getValue() for s in S], 'o', color='blue', label='Returns')
        plt.plot(minis, [Y[s].getValue() for s in minis], 'o', color='red', label='Worst Cases')
        plt.xlabel('Scenarios')
        plt.ylabel('Returns')
        plt.title('Value at Risk Optimization with {:d} scenarios'.format(len(S)))
        plt.legend()
        plt.show()
