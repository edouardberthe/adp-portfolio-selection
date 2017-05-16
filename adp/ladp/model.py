from abc import ABCMeta, abstractmethod

from gurobi import GRB, Model, quicksum
from numpy import array

from adp.parameters import theta
from adp.transition import ft
from data import N


class ADPModel(Model):

    __metadata__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setParam('OutputFlag', False)
        self._x = array([self.addVar(lb=0) for i in range(N)])
        self._y = array([self.addVar(lb=0) for i in range(N)])
        self.update()
        self._holdingConstrs = [self.addConstr(- self._x[i] + self._y[i] <= 0) for i in range(N)]
        self._budgetConstr = self.addConstr(
            (1+theta) * quicksum(self._x) - (1-theta) * quicksum(self._y) <= 0
        )
        self._h = None
        self._h_plus = None

    def set(self, R, h_plus, V):
        self._R = R
        self._h = R * h_plus
        self._V = V
        for i in range(N):
            self._holdingConstrs[i].RHS = self._h[i+1]
        self._budgetConstr.RHS = self._h[0]
        self._h_plus = ft(self._h, self._x, self._y)

    @abstractmethod
    def setADPObjective(self, V):
        raise NotImplementedError

    def solve(self, R, h_plus, V):
        self.set(R, h_plus, V)
        self.setADPObjective(V)
        self.optimize()

    @property
    def x(self):
        return array([v.x for v in self._x])

    @property
    def y(self):
        return array([v.x for v in self._y])

    @property
    def h_plus(self):
        return array([v.getValue() for v in self._h_plus])

    @property
    def ΔV(self):
        return self._V(self._R) + array([self._budgetConstr.Pi] + [cstr.Pi for cstr in self._holdingConstrs]) * self._R


class LADPModel(ADPModel):
    def setADPObjective(self, V):
        super().setObjective(quicksum(V * self._h_plus), GRB.MAXIMIZE)
