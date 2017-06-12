from data import Data
from parameters import periods, theta

for (period, start, middle, end, r) in periods:
    Data_test = Data.loc[middle:end]
    perf = (Data_test.iloc[-1] / Data_test.iloc[0] * (1-theta) - 1).mean()
    print(period, "{:.2%}".format(perf))
