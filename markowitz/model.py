from gurobipy import quicksum

import pandas as pd

from data import A
from entities.model import PortfolioOptimizer


class Markowitz(PortfolioOptimizer):
    """Implements Markowitz model. Extends PortfolioOptimizer, which itself extends gurobipy.Model."""

    def __init__(self, mean: pd.Series, cov: pd.DataFrame, name='Markowitz', *args, **kwargs):
        self._cov = cov
        self._mean = mean
        super().__init__(name, *args, **kwargs)

    def createObjective(self):
        cov = self._cov.as_matrix()
        self.setObjective(
            quicksum(
                cov[i, j] * self._W[i] * self._W[j] for i in A for j in A
            )
        )

    def createConstrs(self):
        super().createConstrs()
        self.addConstr(
            quicksum(self._W[i] * self._mean[i] for i in A) >= self._RRR
    )

    def plot(self):
        pass
