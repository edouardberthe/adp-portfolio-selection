from typing import List

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from numpy import random as rd

from adp.cvar import CVaR
from adp.ladp import LADP_Gurobi
from adp.parameters import *
from adp.transition import ft
from data import *
from linear.scenarios import generateGaussianScenarios

rd.seed(4)

class LinearValueFunction(np.ndarray):
    def __new__(cls):
        return rd.rand(T+1, N+1)


# For each scenario
def step(s, V_hat, h):
    print("Scenario", s)

    # Generating the scenarios
    R = np.exp(np.concatenate((0.005 * np.ones((T+1, 1)), generateGaussianScenarios(T+1)[0]), axis=1))

    Δ_V_tilde = np.zeros((T, N+1))
    V_tilde_plus = np.zeros(N+1)

    print("\tTime 0")
    x, y = LADP_Gurobi(h[s], V_hat[0], np.ones(N+1))
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
    cvar = CVaR((R[T] * h[:s+1]))
    V_hat[T] = gamma * h[s].sum() - (1 - gamma) * cvar

    print("\t\tComputing ΔCVaR")
    CVaR_hat = np.zeros(N+1)
    for i in range(N+1):
        CVaR_hat[i] = CVaR((R[T] * (h + e[i])[:s+1]))
    Δ_CVaR_hat = CVaR_hat - cvar
    print("\t\tComputing ΔV_tilde")
    Δ_V_tilde[T-1] = gamma * R[T] - (1 - gamma) * Δ_CVaR_hat

    print("\tUpdating V_hat[0:T-1]\n")
    V_hat[:T] = (1 - alpha[s]) * V_hat[:T] + alpha[s] * Δ_V_tilde


def update_lines(lines: List[Line3D], V_hat):
    for (i, line) in enumerate(lines):
        line.set_data(xs, i * ys)
        line.set_3d_properties(V_hat[:, i])
    return lines


def run(s, lines: List[Line3D], V_hat, h):
    print('Launched run with s =', s)
    step(s, V_hat, h)
    return update_lines(lines, V_hat)

if __name__ == '__main__':
    # Canonical Basis of R^(N+1)
    e = np.identity(N + 1)

    # Approximate value function
    V_hat = LinearValueFunction()

    # States (stored to compute the CVaR on all terminal states from the beginning)
    h = np.zeros((S, N + 1))
    h[:, 0] = init

    # Animated Plotting figure
    fig = plt.figure(figsize=(20, 10))
    ax = Axes3D(fig)
    # ax.set_ylim(-1.1, 1.1)
    # ax.set_xlim(0, 5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Asset')
    ax.set_zlabel('Value function')
    ax.set_title('Linear Approximate Value Function')

    xs = range(T + 1)
    ys = np.ones(T + 1)

    colors = 'bgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcm'
    lines = []
    lines.append(ax.plot(xs, 0 * ys, V_hat[:, 0], label='Cash', color=colors[0])[0])
    for i in range(1, N):
        lines.append(ax.plot(xs, i * ys, V_hat[:, i], label=Data.columns[i - 1], color=colors[i])[0])
    ani = FuncAnimation(fig, run, S, blit=True, fargs=(lines, V_hat, h), repeat=False)
    plt.show()
