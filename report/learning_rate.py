import pickle

from parameters import perf_dir_name, init
from report.config import *

S = 4000
k = 100
m = 5
gamma = 0.6
key = 'UU'
type = 'gaussian'

figure(figsize=figsize)
for k in (100, 500, 1000):
    dir_name = perf_dir_name.format(type, S, k, key, m, gamma*10)
    with open(dir_name + 'test', 'rb') as file:
        result = pickle.load(file)
        scenarios = sorted(result.keys())
        perfs = np.array([result[s] for s in scenarios]) / init - 1
        plot(scenarios, perfs, label="k = {:d}".format(k))
xlabel('Scenarios')
ylabel('Test Performance')
title('Period Up - Up, $\gamma$ = 0.6, m = 5, Gaussian scenarios')
gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}\%'.format(y * 100)))
legend()
show()
