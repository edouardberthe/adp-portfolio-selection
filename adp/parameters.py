import numpy as np
from numpy import ones, arange

init = 1e6     # Initial amount
beta = 0.90    # Beta in CVaR
theta = 0.05   # Transaction costs
gamma = 0.5    # Risk aversion
S = 1000       # Samples of training
T = 50         # Time steps (Number of re-balancing periods)
r = 0.0005     # Mean money return

# LADP UB Specific Params
w = 0.5       # Max investment fraction
w0 = w * init

# Step size for updating value function
# alpha = np.concatenate((1 * ones(0), 0.5 / np.sqrt(np.arange(1, S+1))))
alpha = 1 / np.log(arange(3, S+3))

# For Piecewise Linear Approximation, grid
M = 100
a = np.linspace(0, w0, M+1)
da = w0 / M
