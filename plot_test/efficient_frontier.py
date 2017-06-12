from gurobipy import *

import numpy as np
from matplotlib import pyplot as plt

from data import A, CovReturns, MeanReturns, VarReturns


def MaxReturn(MaxVol):
    Stocks = MeanReturns.index
    CovMatrix = CovReturns.as_matrix()
    S = range(len(Stocks))
    m = Model()
    m.setParam('OutputFlag', False)
    W = [m.addVar(name=name, lb=0, ub=1) for name in Stocks]
    m.update()
    m.setObjective(quicksum(W[i] * MeanReturns[i] for i in A), GRB.MAXIMIZE)
    m.addConstr(
        quicksum(CovMatrix[i,j] * W[i] * W[j]
        for i in S for j in S) <= MaxVol)
    m.addConstr(quicksum(W[s] for s in S) == 1)
    m.optimize()

    try:
        Weights = np.array([var.x for var in W])
        print("Expected Return:", "{:.2f} %".format(m.objVal * 100))
        vol = Weights.dot(CovReturns).dot(Weights)
        print("Expected Volatility:", "{:.2f} %".format(vol * 100))
        return Weights, m.objVal, vol
    except GurobiError:
        return None, None, None


def TestMaxReturn():
    MaxVols = np.arange(0.0001, 0.06, 0.0001)
    returns = []
    vols = []
    for MaxVol in MaxVols:
        weights, ret, vol = MaxReturn(MaxVol)
        returns.append(ret)
        vols.append(vol)
    plt.plot(MaxVols, returns, 'blue')
    plt.plot(MaxVols, vols, 'red')
    plt.plot(VarReturns, MeanReturns, 'o')
    plt.show()
    plt.plot(vols, returns)
    plt.show()


def MinVol(RRR):
    S = range(len(MeanReturns))
    CovMatrix = CovReturns.as_matrix()
    m = Model()
    m.setParam('OutputFlag', False)
    W = [m.addVar(name=MeanReturns.index[i], lb=0, ub=1) for i in S]
    m.update()
    m.setObjective(quicksum(CovMatrix[i,j] * W[i] * W[j] for i in S for j in S))
    m.addConstr(quicksum(W[i] * MeanReturns[i] for i in S) >= RRR)
    m.addConstr(quicksum(W[i] for i in S) == 1)
    m.optimize()

    try:
        Weights = np.array([var.x for var in W])
        print("Expected Volatility:", "{:.2f} %".format(m.objVal * 100))
        ret = (Weights * MeanReturns).sum()
        print("Expected Return:", "{:.2f} %".format(ret * 100))
        return Weights, ret, m.objVal
    except GurobiError:
        None, None, None

    return Weights