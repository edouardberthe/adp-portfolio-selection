import pickle

from data import Data
from parameters import init, periods, theta
from report.config import *

S = 2000
for (period, start, middle, end, r) in periods:
    for m in [3, 5]:

        # Testing data
        Data_test = Data[middle:end]
        Gross_test = (Data.pct_change() + 1)[1:][middle:end]
        Gross_test.insert(0, 'r', 1 + r)

        # Equally-Weighted Reference
        ref = (Data_test.iloc[-1] / Data_test.iloc[0] * (1 - theta) - 1).mean()

        for gamma in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            filename = "pickle/results/{:d}/gamma_{:.1f}_period_{}_m_{:d}/test".format(
                S, gamma, period, m)
            try:
                with open(filename, 'rb') as file:
                    test = pickle.load(file)
            except FileNotFoundError as e:
                print(e)
            else:
                scenarios = sorted(test.keys())
                values = [test[s]/init - 1 for s in scenarios]

                plot(scenarios, values)
                plot([0, scenarios[-1] + 1], [ref, ref])
                text(scenarios[-1] + 10, ref, r'Reference = {:.1}%'.format(ref*100))

                title("Period {:s} $\gamma$ = {}, m = {:d}".format(period, gamma, m))
                xlabel('Scenarios')
                ylabel('Test Performance')

                show()
