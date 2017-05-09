from matplotlib import pyplot as plt
from numpy import ones, random as rd, zeros

from adp.cvar import CVaR
from adp.generator import GaussianGenerator
from adp.parameters import *
from adp.pladp import PLADPInspectionModel
from data import *


def step(s):
    print("Scenario", s)

    delta_V_tidle = zeros((T, N+1))
    k = zeros((T, N))  # Will store the indexes of the slopes to update at each time for each asset

    print("\tTime 0")
    m.solve(ones(N+1), h_plus[s], V_hat[0], u0)
    next_h_plus = m.h_plus
    k[0] = m.k

    # 1 <= t <= T - 1
    for t in range(1, T):
        print("\tTime", t)
        R = generator.generate()
        m.solve(R, next_h_plus, V_hat[t], u0)
        next_h_plus = m.h_plus
        delta_V_tidle[t-1] = m.delta_V_tilde
        k[t] = m.k

    h_plus[s] = next_h_plus    # We store h^+_{T-1}
    R = generator.generate()   # Last Returns
    RT[s] = R                  # We store the last returns for later scenarios

    # t = T
    print("\tLast time: t =", T)

    # Updating V_hat[T]
    cvar = CVaR(RT[:s+1] * h_plus[:s+1])
    V_hat[T] = gamma * R.dot(next_h_plus) - (1-gamma) * cvar

    # Computing delta CVaR
    CVaR_hat = zeros(N+1)
    for i in range(N+1):
        CVaR_hat[i] = CVaR(RT[:s+1] * (h_plus[:s+1] + [e[i]]))
    Δ_CVaR_hat = CVaR_hat - cvar

    # Computing delta V_tilde
    delta_V_tidle[T-1] = gamma * R - (1-gamma) * Δ_CVaR_hat

    print("\tUpdating V_hat[0:T]\n")
    V_hat[:T] = (1-alpha[s]) * V_hat[:T] + alpha[s] * delta_V_tidle[:, 1:].reshape(T, N, 1).repeat(M+1, 2)


rd.seed(4)
e = np.identity(N+1)
m = PLADPInspectionModel()       # Model Solver
generator = GaussianGenerator()  # Scenarios Generator
V_hat = ones((T+1, N, M+1))      # Approximate value function
u0 = 1

# States (stored to compute the CVaR on all terminal states from the beginning)
# At this step, it should be called simply h, but then it will be post-decision variables
h_plus = zeros((S, N+1))
h_plus[:, 0] = init

# RT will store the last returns in each scenarios. It will be useful for computing Δ_CVaR_hat.
RT = zeros((S, N+1))

for s in range(S):
    step(s)

plt.plot(V_hat)