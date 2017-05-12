import numpy as np
from numpy import ones, arange

init = 1e6     # Initial amount
theta = 0.1    # Transaction costs
beta = 0.90    # Beta in CVaR
gamma = 0.5    # Risk aversion
S = 1000       # Samples of training
T = 50         # Time steps (Number of re-balancing periods)
r = 0.0001     # Mean money return

# LADP UB Specific Params
w = 0.5        # Max investment fraction
w0 = w * init

# Step size for updating value function
# alpha = np.concatenate((1 * ones(0), 0.5 / np.sqrt(np.arange(1, S+1))))
k = 100
alpha = k / (k + arange(S))

# For Piecewise Linear Approximation, grid
M = 101
a = np.linspace(0, w0, M)
da = w0 / M

# Drawing
repeat = 5    # Number of training epochs between two plots