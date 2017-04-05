from gurobipy import GRB, Model, quicksum

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import random as rd

from data import A, Data
from linear.scenarios import generateGaussianScenarios

# Data
N = len(A)  # Stocks
S = 50      # Samples of training
T = 10      # Time steps (Number of re-balancing periods)
rd.seed(4)

# Parameters
init = 10e6                    # Initial amount
beta = 0.90                    # Beta in CVaR
theta = 0.005                  # Transaction costs
gamma = 0.4                    # Risk aversion
alpha = 1 / np.arange(1, S+1)  # Step size for updating value function

paths, probas = generateGaussianScenarios(T)


def ft(h: np.ndarray, R: np.ndarray, x: np.ndarray, y: np.ndarray):
    """
    Transition function.
    :param h: Pre-decision state at previous time (h_(t-1))
    :param R: Returns at time t
    :param x: Buys
    :param y: Sales
    :return : h_plus, post-decision state variable
    :rtype  : np.ndarray
    """
    h_plus = np.zeros(N + 1, dtype=object)
    h_plus[1:] = R[1:] * h[1:] + x - y
    h_plus[0] = R[0] * h[0] - (1 + theta) * x.sum() + (1 - theta) * y.sum()
    return h_plus


def LADP(h: np.ndarray, u: np.array, R: np.ndarray):
    """
    Returns the Linear Approximated Dynamic Programming solution by inspection.
    :param h: h^+_(t-1), state variable
    :param u: u_it, old value function slopes
    :param R: Return at t
    :rtype: tuple(np.array, np.array, np.array)
    """
    # Return values
    x = np.zeros(N)
    y = np.zeros(N)

    k = u - (1 + theta) * u[0]                        # Buying slopes
    l = - u + (1 - theta) * u[0]                      # Selling slopes

    # Step 0
    j_star = np.argmax(k) if (k > 0).any() else None  # Best buy
    I = np.all((l > 0, R * h > 0), axis=0)            # Sell assets
    if j_star is not None:
        # Sell-to-Buy assets
        J = np.all(
            (l <= 0, (1-theta)/(1+theta) * k[j_star] + l > 0, R * h > 0),
            axis=0)

    # Step 1: selling Sell assets
    if I.any():
        y[I] = (R * h)[I]
        h[0] = R[0] * h[0] + (1-theta) * y[I].sum()

    # Step 2: if there is a Best Buy asset, buy as much as possible with cash
    if j_star is not None:
        x[j_star] = h[0] / (1+theta)
        h[0] = 0

    # Step 3: if there is a Best-Buy and some Sell-to-Buy assets
    if j_star is not None and J.any():
        y[J] = (R * h)[J]
        x[j_star] += (1-theta)/(1+theta) * y[J].sum()

    return x, y


def LADP_Gurobi(h: np.ndarray, u: np.array, R: np.ndarray):
    m = Model('Stochastic Dynamic Programming')
    m.setParam('OutputFlag', False)
    xg = np.array([m.addVar(lb=0) for i in range(N)])
    yg = np.array([m.addVar(lb=0) for i in range(N)])
    m.update()
    constrs = [m.addConstr(-xg[i] + yg[i] <= R[i+1] * h[i+1]) for i in range(N)]
    constr = m.addConstr(
        (1 + theta) * quicksum(xg) - (1 - theta) * quicksum(yg) <= R[0] * h[0]
    )
    m.update()
    m.setObjective(quicksum(u * ft(h, R, xg, yg)), GRB.MAXIMIZE)
    m.optimize()
    return np.array([v.x for v in xg]), np.array([v.x for v in yg])


def CVaR(h):
    """
    Return the beta-CVaR associated to the terminal states h[0], h[1], ...
    :param h: np.array
    """
    f = h.prod(axis=0)
    f.sort()
    l = np.argmax(np.arange(1, len(f) + 1) / len(f) >= 1 - beta)
    return (f[:l-1].sum() - (l-1) * f[l]) / (S * (1-beta)) + f[l]


def plot(V_hat, s):
    fig = plt.figure(figsize=(20, 10))
    ax = Axes3D(fig)
    xs = range(T+1)
    colors = 'bgrcmykw'
    ax.bar(xs, V_hat[:, 0], zs=0, zdir='y', label='Cash', color=colors[0])
    for i in range(1, N):
        ax.bar(xs, V_hat[:, i], zs=i, zdir='y', label=Data.columns[i-1], color=colors[i])
    ax.set_xlabel('Time')
    ax.set_ylabel('Asset')
    ax.set_zlabel('Value function')
    plt.title('Scenario nr {:d}/{:d}'.format(s, S))
    plt.show()

# Canonical Basis of R^(N+1)
e = np.identity(N+1)

# Linear approximations value functions initialized randomly (to have only one solution at 1st step)
V_hat = rd.rand(T+1, N+1)
# States (stored to compute the CVaR on all terminal states from the beginning)
h = np.zeros((S, N+1))
h[:, 0] = init

# For each scenario
for s in range(S):
    print("Scenario", s)

    # Generating the scenarios
    R = np.exp(np.concatenate((0.005 * np.ones((T+1, 1)), generateGaussianScenarios(T+1)[0]), axis=1))

    Δ_V_tilde = np.zeros((T, N+1))
    V_tilde_plus = np.zeros(N+1)

    print("\tTime 0")
    x, y = LADP_Gurobi(h[s], V_hat[0], np.ones(N+1))
    # xg, yg = LADP(h[s], V_hat[0], np.ones(N))
    h[s] = ft(h[s], np.ones(N+1), x, y)

    # 1 <= t <= T - 1
    for t in range(1, T):
        print("\tTime", t)

        for i in range(N+1):
            x, y = LADP_Gurobi(h[s] + e[i], V_hat[t], R[t])
            V_tilde_plus[i] = V_hat[t].dot(ft(h[s] + e[i], R[t], x, y))

        x, y = LADP_Gurobi(h[s], V_hat[t], R[t])
        h[s] = ft(h[s], R[t], x, y)
        Δ_V_tilde[t-1] = V_tilde_plus - V_hat[t].dot(h[s])

    # t = T
    print("\tLast time")
    print("\t\tUpdating V_hat[T]")
    cvar = CVaR((R[T] * h)[:s+1])
    V_hat[T] = gamma * h[s].sum() - (1 - gamma) * cvar

    print("\t\tComputing ΔCVaR")
    CVaR_hat = np.zeros(N+1)
    for i in range(N+1):
        CVaR_hat[i] = CVaR((R[T] * (h + e[i]))[:s+1])
    Δ_CVaR_hat = CVaR_hat - cvar
    print("\t\tComputing ΔV_tilde")
    Δ_V_tilde[T-1] = gamma * R[T] - (1 - gamma) * Δ_CVaR_hat

    print("\tUpdating V_hat[0:T-1]\n")
    V_hat[:T] = (1 - alpha[s]) * V_hat[:T] + alpha[s] * Δ_V_tilde
    if s % 10 == 0:
        plot(V_hat, s)

plot(V_hat, S)
