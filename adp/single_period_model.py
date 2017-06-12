from gurobipy import GRB, LinExpr, Model, quicksum

import numpy as np

from data import N
from parameters import beta, init, periods, theta


def singlePeriodModel(R: np.ndarray, gamma: float) -> np.ndarray:
    """
    Computes the Single Period Portfolio Selection Problem, with R as return
    samples.
    """
    S = len(R)
    m = Model()
    m.setParam('OutputFlag', False)

    # Variables
    x = np.array([m.addVar(lb=0) for _ in range(N)])
    y = np.array([m.addVar(lb=0) for _ in range(N)])
    g0 = m.addVar(lb=-GRB.INFINITY)
    g2 = [m.addVar(lb=0) for _ in range(S)]

    m.update()

    # Linear Expr
    h = np.zeros(N+1, dtype=LinExpr)
    h[1:] = x - y
    h[0] = init - (1 + theta) * quicksum(x) + (1 - theta) * quicksum(y)

    # Constraints
    [m.addConstr(hv >= 0) for hv in h]
    [m.addConstr(v >= - g0 - quicksum(R[s] * h) + init) for (s, v) in enumerate(g2)]

    m.setObjective(gamma * quicksum(quicksum(r * h) for r in R)
                   - (1 - gamma) * g0
                   - (1 - gamma) / (1 - beta) * quicksum(g2), GRB.MAXIMIZE)

    m.optimize()
    return np.array([v.getValue() for v in h])

if __name__ == '__main__':
    from adp.generator import GaussianGenerator

    from parameters import S

    (period, start, middle, end, r) = periods[0]
    R = GaussianGenerator(r=r, start=start, end=middle).generate(S)
    gamma = 0.2
    h = singlePeriodModel(R, gamma)
