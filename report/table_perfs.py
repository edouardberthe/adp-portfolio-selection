import os
import pickle

import numpy as np

from adp.generator import GaussianGenerator
from adp.single_period_model import singlePeriodModel
from data import Data
from markowitz import Markowitz
from parameters import S, init, periods, theta

for (period, start, middle, end, r) in periods:
    # Training data
    Return_train = Data.pct_change()[1:][start:middle] * 252

    # Testing data
    Data_test = Data[middle:end]
    Gross_test = (Data.pct_change() + 1)[1:][middle:end]
    Gross_test.insert(0, 'r', 1 + r)

    # Equally-Weighted Reference
    ref = (Data_test.iloc[-1] / Data_test.iloc[0] * (1 - theta) - 1).mean()
    print(period, 'reference = {:.1%}'.format(ref))

    print('\tMarkowitz')
    for RRR in np.arange(0.1, 0.5, 0.05):

        model = Markowitz(mean=Return_train.mean(), cov=Return_train.cov(), RRR=RRR)
        port = model.optimize().getPortfolio()
        # port.plot()

        # Testing
        perf = (Data_test.iloc[-1] / Data_test.iloc[0] * port * (1-theta)).sum() - 1
        print('\t\t', '{:.2%} {:.1%}'.format(RRR, perf))

    g = GaussianGenerator(r=r, start=start, end=middle)

    print('\tSingle Period')
    # Scenarios Generator on the training data
    for gamma in [0, 0.2, 0.4, 0.6, 0.8, 1]:

        R = np.array([g.generate() for s in range(S)])
        h = singlePeriodModel(R, gamma)

        perf = ((1+r) * h[0] + (Data_test.iloc[-1] / Data_test.iloc[0] * h[1:]).sum()) / init - 1
        print('\t\t{}: {:.1%}'.format(gamma, perf))

    m = 5
    print('\tMulti Period')
    for gamma in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        dirname = "pickle/results/{:d}/gamma_{:.1f}_period_{}_m_{:d}/".format(
            S, gamma, period.replace(' ', ''), m)
        os.makedirs(dirname, exist_ok=True)
        strategy = pickle.load(file=open(dirname + "strategy", "rb"))
        perf = strategy.score(Gross_test) / init - 1
        print('\t\t{}: {:.1%}'.format(gamma, perf))
