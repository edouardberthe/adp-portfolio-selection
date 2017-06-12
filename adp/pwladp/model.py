from gurobipy import GRB, GurobiError, Model, quicksum

import numpy as np
from numpy import array

from adp.value_function import ValueFunction
from data import N
from parameters import theta


class PWLADPGurobiSolver(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setParam('OutputFlag', False)

        # Variables
        self._hpv = array([self.addVar(lb=-GRB.INFINITY) for i in range(N)])  # h_plus at time t
        self._xv = array([self.addVar() for i in range(N)])   # Buys at time t
        self._yv = array([self.addVar() for i in range(N)])   # Sales at time t
        self.update()

        # Linear Expression (input Cash Flow)
        self._outputCashFlow = (1+theta) * quicksum(self._xv) - (1-theta) * quicksum(self._yv)

        # Constraints
        self._eqCstrs = [self.addConstr(self._hpv[i] - self._xv[i] + self._yv[i] == 0) for i in range(N)]
        self._holdingCstrs = [self.addConstr(-self._xv[i] + self._yv[i] <= 0) for i in range(N)]
        self._budgetCstr = self.addConstr(- self._outputCashFlow <= 0)
        self.update()

    def solve(self, R, hp, V):
        h = R * hp

        # Storing
        self._V = V
        self._R = R
        self._h = h

        cash = h[0] - self._outputCashFlow

        # For the moment, we put only the cash value, but we will add
        # the pwl value function of the assets' positions.
        self.setObjective(V.cash * cash, GRB.MAXIMIZE)

        # The y coordinates of the points of the pwl value function
        for i in range(N):
            # h is (N+1) sized
            self._eqCstrs[i].RHS = h[i+1]
            self._holdingCstrs[i].RHS = h[i+1]
            self.setPWLObj(self._hpv[i], V.V[i].x(), V.V[i].y())
        self._budgetCstr.RHS = h[0]

        self.optimize()

        return np.around(np.maximum([cash.getValue()] + [v.x for v in self._hpv], 0), decimals=0), \
               array([self._budgetCstr.Pi] + [c.Pi for c in self._holdingCstrs]) * R


def gurobiModel(R, hp, V: ValueFunction) -> (np.ndarray, np.ndarray):
    """
    Little precision: x and y have no lower bound during declaration, because else holdingCstrs have no dual variables.
    :param R:    Returns at time t (Gross)
    :param hp:   Post-decision variable (pre-return) at time t-1
    :param V:    Value function at time t
    :return:     Post-decision variable at time t + DeltaV
    """
    h = R * hp

    m = Model()
    m.setParam('OutputFlag', False)

    # Variables
    hpv = array([m.addVar(lb=-GRB.INFINITY) for _ in range(N)])  # h_plus at time t
    xv = array([m.addVar() for _ in range(N)])                   # Buys   at time t
    yv = array([m.addVar() for _ in range(N)])                   # Sales  at time t
    m.update()

    # Linear Expressions
    outputCashFlow = (1+theta) * quicksum(xv) - (1-theta) * quicksum(yv)
    """:type: gurobipy.LinExpr"""

    # Objective
    m.setObjective(- V.cash * outputCashFlow, GRB.MAXIMIZE)
    for i in range(N):
        m.setPWLObj(hpv[i], V[i].x(), V[i].y())

    # Constraints
    eqCstrs = [m.addConstr(hpv[i] - h[i+1] == xv[i] - yv[i]) for i in range(N)]
    holdingCstrs = [m.addConstr(- xv[i] + yv[i] <= h[i+1]) for i in range(N)]
    budgetCstr = m.addConstr(outputCashFlow <= h[0])

    try:
        m.optimize()
    except GurobiError as e:
        pass

    return np.maximum([h[0] - outputCashFlow.getValue()] + [v.x for v in hpv], 0), \
           V(R) + array([budgetCstr.Pi] + [c.Pi for c in holdingCstrs]) * R
