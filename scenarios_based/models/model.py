from time import time

from gurobi import quicksum

from data import A
from entities.model import PortfolioOptimizer


class ScenariosBasedPortfolioModel(PortfolioOptimizer):
    """
    Overrides entities.portfolio.PortfolioOptimizer class. A scenarios_based portfolio model must have a list of possible scenarios
    with probabilities.
    """

    def __init__(self, name, scenarios=None, probas=None, *args, **kwargs):
        """
        Creates a Linear Portfolio Model.
        :type scenarios: numpy.array - Return of each stock in each scenario
        :type probas:    numpy.array - Probability of each scenario
        """
        # We store the sets
        self._scenarios = scenarios
        self._probas = probas
        self._S = range(len(scenarios))
        
        # Call to PortfolioOptimizer constructor
        super().__init__(name, *args, **kwargs)

    def createVars(self):
        super().createVars()
        # We do not need specific variables for this model, all we need to do is creating the helpers LinExpr.
        self.createLinExpr()

    def createLinExpr(self):
        # Mean return of each stock
        MuA = self._probas.dot(self._scenarios)

        # Mean return of the portfolio (gurobipy.LinExpr)
        self._mu = quicksum(MuA[a] * self._W[a] for a in A)

        # Return in each scenario (gurobipy.LinExpr)
        self._Y = [
            quicksum(self._scenarios[s, a] * self._W[a] for a in A)
            for s in self._S
        ]

    def createConstrs(self):
        """Adds the base constraints of the Linear Model portfolio."""
        super().createConstrs()
        self._RRRConstr = self.addConstr(self._mu * 252 >= self._RRR)

    def reconfigure(self, scenarios, probas):
        """
        :rtype: ScenariosBasedPortfolioModel
        """
        # Stores the new scenarios and probas
        self._scenarios = scenarios
        self._probas = probas

        # Removes the internal LinExpr _mu and _Y (note: this is not 'useful', because in the following lines the
        # constraints will be updated through the method chgCoeff, but this is only for sake of consistency, to avoid
        # having internal data unconsistent with the constraints).
        self._mu = None
        self._Y = None

        MuA = probas.dot(scenarios)

        # Updates the Required Rate of Return constraint
        if self._output:
            t = time()
            print("Updating RRR Constr")
        [self.chgCoeff(self._RRRConstr, self._W[a], 252 * MuA[a]) for a in A]
        if self._output:
            print("\t{:.1f} s".format(time() - t))

        return self