import pickle

from data import Data
from parameters import init, periods, theta, perf_dir_name
from report.config import *


S = 4000
k = 500
type = 'gaussian'
key = 'DD'
(period, start, middle, end, r) = periods[key]
m = 5

# Testing data
Data_test = Data[middle:end]
Gross_test = (Data.pct_change() + 1)[1:][middle:end]
Gross_test.insert(0, 'r', 1 + r)

# Equally-Weighted Reference
ref = (Data_test.iloc[-1] / Data_test.iloc[0] * (1 - theta) - 1).mean()

for gamma in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    file_name = perf_dir_name.format(type, S, k, key, m, gamma * 10) + "test"
    try:
        with open(file_name, 'rb') as file:
            test = pickle.load(file)
    except FileNotFoundError as e:
        print(e)
    else:
        scenarios = sorted(test.keys())
        values = [test[s]/init - 1 for s in scenarios]

        figure(figsize=figsize)
        plot(scenarios, values)
        plot([0, S], [ref, ref])
        text(scenarios[-1] + 100, ref, r'EW Perf {:.1f} \%'.format(ref*100))

        title("Period {:s} $\gamma$ = {}, m = {:d}".format(period, gamma, m))
        xlabel('Scenarios')
        xlim(0, S)
        ylabel('Test Performance')
        gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}\%'.format(y * 100)))

        subplots_adjust()
        show()
