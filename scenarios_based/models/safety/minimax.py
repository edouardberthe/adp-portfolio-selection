from gurobipy import GRB
from time import time

from matplotlib import pyplot as plt

from data import A
from scenarios_based.models.model import ScenariosBasedPortfolioModel
from generator import generate_gaussian


class Minimax(ScenariosBasedPortfolioModel):
    """Implements Minimax model. Extends ScenariosBasedPortfolioModel, which itself extends gurobipy.Model."""

    def __init__(self, scenarios, probas, name='Minimax', *args, **kwargs):
        super().__init__(name, scenarios, probas, *args, **kwargs)

    def createVars(self):
        """Adds the variables specific to the Minimax model."""
        super().createVars()
        self._My = self.addVar(lb=-GRB.INFINITY)

    def createObjective(self):
        self.setObjective(self._My, GRB.MAXIMIZE)

    def createConstrs(self):
        super().createConstrs()
        self._cstr = [
            self.addConstr(self._Y[s] >= self._My)
            for s in self._S
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

        if self._output:
            t = time()
            print("Updating Constr")
        for a in A:
            for s in self._S:
                self.chgCoeff(self._cstr[s], self._W[a], scenarios[s, a])
        if self._output:
            print("\t{:.2f} s".format(time() - t))

        return self

    def plot(self):
        S, mu, Y = self._S, self._mu, self._Y

        # Plots the Minimax
        minis = [s for s in S if abs(Y[s].getValue() - self.objVal) < 10e-5]

        plt.plot(S, [Y[s].getValue() for s in S], 'o', color='blue',
                 label='Returns')
        plt.plot(minis, [Y[s].getValue() for s in minis], 'o', color='red',
                 label='Worst Cases')
        plt.plot([S[0], S[-1]], [self.objVal, self.objVal], 'red')
        plt.xlabel('Scenarios')
        plt.ylabel('Returns')
        plt.title('Minimax Optimization with {:d} scenarios'.format(len(S)))
        plt.legend()
        plt.show()

if __name__ == '__main__':
    s, p = generate_gaussian(200)
    Minimax(s, p, output=True).optimize().plot()
