from gurobipy import GRB, Model, quicksum

import numpy as np

from data import N
from .parameters import *
from .transition import ft


def LADP(h_plus: np.ndarray, u: np.array, R: np.ndarray):
    """
    Returns the Linear Approximated Dynamic Programming solution by inspection.
    :param h_plus: h^+_(t-1), state variable
    :param u     : u_it, old value function slopes
    :param R     : Return at t
    :rtype: tuple(np.array, np.array, np.array)
    """

    # Computing pre-decision variables
    h = h_plus * R

    # Return values
    x = np.zeros(N)
    y = np.zeros(N)

    k = u[1:] - (1 + theta) * u[0]                        # Buying slopes
    l = - u[1:] + (1 - theta) * u[0]                      # Selling slopes

    # Step 0
    j_star = np.argmax(k) if (k > 0).any() else None  # Best buy
    I = np.all((l > 0, h[1:] > 0), axis=0)            # Sell assets
    if j_star is not None:
        # Sell-to-Buy assets
        J = np.all(
            (l <= 0, (1-theta)/(1+theta) * k[j_star] + l > 0, h[1:] > 0), axis=0)

    # Step 1: selling Sell assets
    if I.any():
        y[I] = h[1:][I]
        h[0] += (1-theta) * y[I].sum()

    # Step 2: if there is a Best Buy asset, buy as much as possible with cash
    if j_star is not None:
        x[j_star] = h[0] / (1+theta)
        h[0] = 0

    # Step 3: if there is a Best-Buy and some Sell-to-Buy assets
    if j_star is not None and J.any():
        y[J] = h[1:][J]
        x[j_star] += (1-theta)/(1+theta) * y[J].sum()

    return x, y

m = Model('Stochastic Dynamic Programming')
m.setParam('OutputFlag', False)
xg = np.array([m.addVar(lb=0) for i in range(N)])
yg = np.array([m.addVar(lb=0) for i in range(N)])
m.update()
constrs = [m.addConstr(-xg[i] + yg[i] <= 0) for i in range(N)]
constr = m.addConstr(
    (1 + theta) * quicksum(xg) - (1 - theta) * quicksum(yg) <= 0
)
m.update()


def LADP_Gurobi(h: np.ndarray, u: np.array, R: np.ndarray):
    for i in range(N):
        constrs[i].RHS = R[i+1] * h[i+1]
    constr.RHS = R[0] * h[0]
    m.setObjective(quicksum(u * ft(h, R, xg, yg)), GRB.MAXIMIZE)
    m.optimize()
    return np.array([v.x for v in xg]), np.array([v.x for v in yg])
