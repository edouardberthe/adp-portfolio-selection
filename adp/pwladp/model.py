from gurobipy import GRB, quicksum

import numpy as np
from numpy import array

from adp.ladp import ADPModel
from adp.value_function import PWLinearValueFunction
from data import N


class PWLADPModel(ADPModel):

    def set(self, R, h_plus, V: PWLinearValueFunction):
        super().set(R, h_plus, V)
        k = V.buying_y()
        l = V.selling_y()
        for i in range(N):
            self.setPWLObj(self._x[i], V.a[:, i], k[:, i])
            self.setPWLObj(self._y[i], V.a[:, i], l[:, i])

    def setADPObjective(self, V):
        self.setObjective(quicksum(self._x) + quicksum(self._y), GRB.MAXIMIZE)

    @property
    def pi(self):
        return np.argmin(self._V.idx(self._h[1:]), axis=0) - 1

    @property
    def ΔV(self):
        return array([self._budgetConstr.Pi] + [cstr.Pi for cstr in self._holdingConstrs]) * self._R
