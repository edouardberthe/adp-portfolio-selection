import numpy as np

init = 1e5                           # Initial amount
theta = 0.002                        # Transaction costs
beta = 0.95                          # Beta in CVaR
gammas = [0, 0.2, 0.4, 0.6, 0.8, 1]  # Risk aversion
S = 4000                             # Samples of training
T = 52                               # Time steps (Number of re-balancing periods)
freq = 'W-FRI'                       # Frequency of rebalancing

# LADP UB Specific Params
w = 0.5                              # Max investment fraction
w0 = w * init

# Step size for updating value function
k = 500
alpha = lambda s: k / (k + s)

# For Piecewise Linear Fixed Approximation, grid
M = 11
a = np.linspace(0, init, M)

# For Dynamic PWL Value Functions
decimals = -3   # Rounding
m = 5           # Max number of slopes

# Plotting
repeat = 5      # Number of training epochs between two plots
figsize = (8, 4)

# Training period
train_start = None
train_end = '2015'
test_start = '2016'
test_end = '2016'

periods = {
    'DD': ('Down - Down', '2006-06-30', '2008-06-27', '2009-06-26', 0.00031),
    'DU': ('Down - Up',   '2007-03-09', '2009-03-06', '2010-03-05', 0.00030),
    'UD': ('Up - Down',   '2013-05-03', '2015-05-01', '2016-04-29', 0.00012),
    'UU': ('Up - Up',     '2012-06-01', '2014-05-30', '2015-05-29', 0.00014)
}

perf_dir_name = "pickle/results/{:s}/{:d}_{:d}/{:s}/m_{:d}/gamma_{:.0f}/"
fig_file_name = "report/figures/perfs/{:s}/gamma_{:d}"
