from gurobipy import Model, quicksum

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from data import CovReturns, MeanReturns, VarReturns, VolReturns, figsize
from markowitz import Markowitz


def plotVolReturns():
    plt.figure(figsize=figsize)

    plt.plot(VolReturns, MeanReturns, 'o')

    plt.title('CAC 40 stocks\' annualized returns')
    plt.xlabel('Volatility (standard deviation)')
    plt.ylabel('Mean')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    mr, Mr, mv, Mv = min(MeanReturns), max(MeanReturns), min(VolReturns), max(VolReturns)

    ax = plt.gca()
    ax.text((Mv + mv)/2, mr, 'Riskier',   ha="center", va="center", size=18,
            bbox=dict(boxstyle="rarrow,pad=0.3", fc="red", ec="b", lw=2))
    ax.text(mv, (Mr + mr)/2, 'Profitable', ha="center", va="center", size=18, rotation=90,
            bbox=dict(boxstyle="rarrow,pad=0.3", fc="green", ec="b", lw=2))

    plt.show()


def plot2StocksPortfolio():
    s1 = 'FR.PA'
    s2 = 'ENGI.PA'

    lamb = np.linspace(0, 1, 1000)
    means = lamb * MeanReturns[s1] + (1 - lamb) * MeanReturns[s2]
    vars = lamb ** 2 * VarReturns[s1] \
           + (1 - lamb) ** 2 * VarReturns[s2] \
           + lamb * (1 - lamb) * CovReturns.loc[s1, s2]

    plt.figure(figsize=figsize)
    plt.plot(VarReturns, MeanReturns, 'o')
    plt.plot(VarReturns[[s1, s2]], MeanReturns[[s1, s2]], 'o', color='red')
    plt.plot(vars, means, '-')

    plt.title('Possible Portfolios with {:s} and {:s}'.format(s1, s2))
    plt.xlabel('Returns Variance')
    plt.ylabel('Returns Mean')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax = plt.gca()
    for c in [s1, s2]:
        ax.annotate(c, xy=(VarReturns[c], MeanReturns[c]), xytext=(VarReturns[c] + 0.001, MeanReturns[c] - 0.02))

    plt.show()


def plot3StocksPortfolios():
    s = ['FR.PA', 'ENGI.PA', 'BNP.PA']
    N = range(len(s))
    M = [(i, j) for i in N for j in N if i < j]

    lamb = np.linspace(0, 1, 1000)
    means = [
        lamb * MeanReturns[s[i]] + (1 - lamb) * MeanReturns[s[j]]
        for i, j in M
    ]
    vars = [
        lamb ** 2 * VarReturns[s[i]] \
        + (1 - lamb) ** 2 * VarReturns[s[j]] \
        + 2 * lamb * (1 - lamb) * CovReturns.loc[s[i], s[j]]
        for i, j in M
    ]

    m = Model()
    m.setParam('OutputFlag', False)
    Lamb = [m.addVar(lb=0, ub=1) for i in N]
    m.update()
    m.setObjective(quicksum(Lamb[i] * Lamb[j] * CovReturns.loc[s[i], s[j]] for i in N for j in N))
    m.addConstr(quicksum(Lamb) == 1)
    ReturnConstr = m.addConstr(quicksum(Lamb[i] * MeanReturns[s[i]] for i in N) == 0)
    m.update()

    OptMean = []
    OptVar = []
    for ret in np.linspace(min(MeanReturns[s]), max(MeanReturns[s]), 1000):
        ReturnConstr.rhs = ret
        m.optimize()
        try:
            OptVar.append(m.objVal)
            OptMean.append(ret)
        except Exception:
            pass

    plt.figure(figsize=figsize)
    plt.plot(VarReturns, MeanReturns, 'o', color='blue')
    plt.plot(VarReturns[s], MeanReturns[s], 'o', color='red')
    for i in N:
        plt.plot(vars[i], means[i], '-', color='green', label='2 stocks portfolios' if i == 0 else None)
    plt.plot(OptVar, OptMean, '-', color='red', label='Gurobi optimization')

    plt.title('Possible Portfolios with {:s}, {:s} and {:s}'.format(*s))
    plt.xlabel('Returns Variance')
    plt.ylabel('Returns Mean')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.legend(loc='upper left')
    ax = plt.gca()
    for c in s:
        ax.annotate(c, xy=(VarReturns[c], MeanReturns[c]), xytext=(VarReturns[c] + 0.001, MeanReturns[c] - 0.02))

    plt.show()


def plotCAC40(riskFree=None, **kwargs):
    """
    Plots the efficient frontier, computed by Markowitz Portfolio Optimization.
    :param riskFree: float|none
    :param kwargs:   argument to pass to Markowitz Portfolio Model
    """
    MinReturns = np.linspace(MeanReturns.min(), MeanReturns.max(), 100)
    returns = []
    variances = []
    for RRR in MinReturns:
        port = Markowitz(RRR=RRR, **kwargs).optimize().getPortfolio()
        returns.append(port.MeanReturn())
        variances.append(port.Vol() ** 2)
        print("{:.1%} return, {:.1%} vol".format(port.MeanReturn(), port.Vol()))

    plt.figure(figsize=figsize)
    plt.plot(VarReturns, MeanReturns, 'o', color='blue')
    plt.plot(variances, returns, color='green')
    plt.title('CAC 40 - Efficient frontier')
    plt.xlabel('Returns Variance')
    plt.ylabel('Returns Mean')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    if riskFree is not None:
        ind = int(0.75 * len(MinReturns))
        plt.plot([0, vols[ind]], [riskFree, returns[ind]], 'o-', color='red')
    plt.show()


if __name__ == '__main__':
    plotVolReturns()
    plot2StocksPortfolio()
    plot3StocksPortfolios()
    plotCAC40()
