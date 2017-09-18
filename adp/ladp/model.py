from abc import ABCMeta, abstractmethod

from gurobipy import GRB, Model, quicksum
from numpy import array, identity

from adp.transition import ft
from data import N
from parameters import theta


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
        self._h_plus_pre = h_plus
        self._h = R * h_plus
        self._V = V
        for i in range(N):
            self._holdingConstrs[i].RHS = self._h[i+1]
        self._budgetConstr.RHS = self._h[0]
        self._h_plus = ft(self._h, self._x, self._y)

    @abstractmethod
    def setADPObjective(self, V):
        raise NotImplementedError

    def solve(self, R, h_plus, V) -> Model:
        self.set(R, h_plus, V)
        self.setADPObjective(V)
        self.optimize()
        return self

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
    def deltaV(self):
        return self._V(self._R) + array([self._budgetConstr.Pi] + [cstr.Pi for cstr in self._holdingConstrs]) * self._R

    def manualΔV(self):
        e = identity(N+1)
        V_plus = array([
            self.solve(self._R, self._h_plus_pre + e[i], self._V).objVal
            for i in range(N+1)
            ])
        return V_plus - self.solve(self._R, self._h_plus_pre, self._V).objVal


class LADPModel(ADPModel):
    def setADPObjective(self, V):
        self.setObjective(quicksum(V * self._h_plus), GRB.MAXIMIZE)
