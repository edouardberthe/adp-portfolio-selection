from time import time

from gurobi import GRB, GurobiError, Model, quicksum

from data import A, Data
from .portfolio import EquallyWeightedPortfolio, Portfolio


class PortfolioOptimizer(Model):
    """
    Extend gurobi.Model class. Doing this allows more flexibility and reusability of the class. Furthermore,
    inheritance makes all the methods of gurobipy.Model also available here.
    """

    def __init__(self, name, output=False, RRR=0.09, Nmax=None, Wmin=None, Wmax=1):
        """
        Creates a Portfolio Model.
        :type name:      string       - Name of the Model
        :type RRR:       float        - Annualized Required Rate of Return
        :type output:    bool  | None - Passed to OutputFlag Gurobi param
        :type Nmax:      int   | None - Max number of stocks to invest in. If None, there is no limit.
        :type Wmin:      float | None - Min investment in a stock. If None, there is no minimum investment.
        :type Wmax:      float        - Max investment in a stock (%) (default 1.0).

        TODO:
        :type leverage:  bool  | None - Leverage to reach (ratio long/short). If None, we cannot short a stock.
        """
        modelName = '{:s}{:s}{:s}{:s}'.format(
            name,
            ', Nmax = {:d}'.format(Nmax) if Nmax is not None else '',
            ', Wmin = {:.1%}'.format(Wmin) if Wmin is not None else '',
            ', Wmax = {:.1%}'.format(Wmax) if Wmax < 1 else '',
        )
        # Call to gurobipy.Model constructor
        super().__init__(name=modelName)
        if not output:
            self.setParam('OutputFlag', False)

        # We select the data


        # We store the params
        self._Nmax = Nmax
        self._Wmin = Wmin
        self._Wmax = Wmax
        self._RRR = RRR
        self._output = output

        if output:
            print("\n\n####### Creating model {:s} #######\n".format(modelName))
            t = time()
            print("Adding Variables")
        self.createVars()
        if output:
            print("\t{:.2f} s".format(time() - t))
            t = time()
            print("Updating")
        self.update()
        if output:
            print("\t{:.2f} s".format(time() - t))
            t = time()
            print("Setting Objective")
        self.createObjective()
        if output:
            print("\t{:.2f} s".format(time() - t))
            t = time()
            print("\n\n####### Creating Constraints ########\n")
        self.createConstrs()
        if output:
            print("\t{:.2f} s".format(time() - t))
            print("Done\n")

    def createVars(self):
        # Weights
        self._W = [self.addVar(name=Data.columns[a], lb=0, ub=self._Wmax) for a in A]

        if self._Nmax is not None or self._Wmin is not None:
            # Do we invest in X?
            self._X = [self.addVar(vtype=GRB.BINARY) for a in A]

    def createConstrs(self):
        """
        Adds the base constraints of the Linear Model portfolio. We could have put it in the '__init__' but then we
        would have had to do an 'update' in the '__init__' and another 'update' for each child of this class.
        """
        self.addConstr(quicksum(self._W) == 1)
        if self._Nmax is not None or self._Wmin is not None:
            [self.addConstr(self._W[a] <= self._X[a]) for a in A]
        if self._Nmax is not None:
            self.addConstr(quicksum(self._X) <= self._Nmax)
        if self._Wmin is not None:
            [self.addConstr(self._W[a] >= self._X[a] * self._Wmin) for a in A]

    def getPortfolio(self):
        """
        Return an object Portfolio generated from the result of the algorithm. Must be used after 'optimize'.
        :rtype Portfolio
        """
        try:
            return Portfolio(data=[w.x for w in self._W], index=Data.columns, name=self.ModelName)
        except GurobiError:
            # If the model is infeasible, we return an equally-weighted portfolio.
            return EquallyWeightedPortfolio()

    def optimize(self):
        """
        Extends gurobipy.Model.optimize to return the Model object for practical reasons. Allows chaining commands as
        for instance:
            MAD(s,p).optimize().getPortfolio()
        :rtype: PortfolioOptimizer
        """
        super().optimize()
        return self

    def update(self):
        """
        Extends gurobipy.Model.update to return the Model object for practical reasons. Allows chaining commands as
        for instance:
            MAD(s,p).update().reconfigure(s2, p2).optimize()
        :rtype: PortfolioOptimizer
        """
        super().update()
        return self


class EWPortfolioModel(object):
    """
    This model has the same interface as a PortfolioOptimizer
    """

    def __init__(self, *args, **kwargs):
        """Does not do anything."""
        super().__init__()
        self.objval = None

    def getPortfolio(self):
        return EquallyWeightedPortfolio()

    def update(self, *args, **kwargs):
        pass

    def reconfigure(self, *args, **kwargs):
        pass

    def optimize(self, *args, **kwargs):
        pass

