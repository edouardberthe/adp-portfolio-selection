from gurobipy import GRB, GurobiError, Model, quicksum

import numpy as np
from numpy import array

from adp.value_function import ValueFunction
from data import N
from parameters import theta


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
