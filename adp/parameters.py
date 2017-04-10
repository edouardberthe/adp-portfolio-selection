import numpy as np


init = 10e6    # Initial amount
beta = 0.90    # Beta in CVaR
theta = 0.005  # Transaction costs
gamma = 0.4    # Risk aversion
S = 100        # Samples of training
T = 50         # Time steps (Number of re-balancing periods)

# Step size for updating value function
alpha = 1 / np.sqrt(np.arange(1, S+1))
