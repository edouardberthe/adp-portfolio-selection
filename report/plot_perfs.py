import os
import pickle

from report.config import *
from matplotlib.ticker import FuncFormatter

from adp.generator import GaussianGenerator
from adp.single_period_model import singlePeriodModel
from data import Data
from markowitz import Markowitz
from parameters import init, periods, theta, gammas, perf_dir_name, fig_file_name

figsize = (8, 3)
RRR = 0.2
S = 4000
k = 500
for (key, (period, start, middle, end, r)) in periods.items():

    # Training data
    Return_train = Data.pct_change()[1:][start:middle] * 252
    """:type: pandas.DataFrame"""


    # Testing data
    Data_test = Data[middle:end]
    Gross_test = (Data.pct_change() + 1)[1:][middle:end]
    Gross_test.insert(0, 'r', 1 + r)
    Weekly_Gross_test = (Data.asfreq('W-FRI', method='pad').pct_change()[1:] + 1)[middle:end]
    Weekly_Gross_test.insert(0, 'r', 1 + r)


    # Equally-Weighted Reference
    perf_ew = (Gross_test.cumprod().mean(axis=1) * (1 - theta) - 1)
    """:type: pandas.DataFrame"""
    print(period)
    print('Equally-Weighted reference = {:.1%}'.format(perf_ew[-1]))


    # Markowitz
    perf_marko = {}
    for RRR in (0.1, 0.2, 0.3):
        model = Markowitz(mean=Return_train.mean(), cov=Return_train.cov(), RRR=RRR)
        port = model.optimize().getPortfolio()
        perf_marko[RRR] = (Gross_test.iloc[:, 1:].cumprod() * port).sum(axis=1) * (1 - theta) - 1
        """:type: pandas.DataFrame"""
        print('Markowitz {}: {:.1%}'.format(RRR, perf_marko[RRR][-1]))

    for gamma in gammas:
        print('Gamma', gamma)

        fig = figure(figsize=figsize)
        ax = fig.gca()

        ax.plot([Data_test.index[0], Data_test.index[-1]], [0, 0], color='black', lw=1)
        perf_ew.plot(ax=ax, label='EW Portfolio', color='blue', lw=1)
        colors = ['purple', 'cyan', 'yellow']
        for i, (RRR, perf_RRR) in enumerate(perf_marko.items()):
            perf_RRR.plot(ax=ax, label='Markowitz {:.0f}\%'.format(RRR*100), lw=1, color=colors[i])

        # Single Period
        R = GaussianGenerator(r=r, start=start, end=middle).generate(S)
        single_h = singlePeriodModel(R, gamma)
        perf_single = (Gross_test.cumprod() * single_h).sum(axis=1) / init - 1
        """:type: pandas.DataFrame"""
        print('\tSingle {}: {:.1%}'.format(gamma, perf_single[-1]))
        perf_single.plot(ax=ax, label='SP', color='green', lw=1)

        # Multi Period
        for (type, type_key) in (('gaussian', 'G'), ('student', 'T')):
            for (m, color) in [(3, 'orange'), (5, 'red')]:
                dir_name = perf_dir_name.format(type, S, k, key, m, gamma*10)
                os.makedirs(dir_name, exist_ok=True)
                try:
                    with open(dir_name + "strategy", "rb") as file:
                        strategy = pickle.load(file)
                except FileNotFoundError as e:
                    print(e)
                else:
                    multi_perf = strategy.score(Weekly_Gross_test).sum(axis=1) / init - 1
                    multi_perf.name = 'PWL ADP {:s} m = {:d}'.format(type_key, m)
                    """:type: pandas.DataFrame"""
                    print('\tMulti Period, {:s}, m={:d}: {:.1%}'.format(type, m, multi_perf.iloc[-1]))
                    multi_perf.plot(ax=ax, lw=1, color=color)

        ax.set_title('Performance {:s}, $\gamma$ = {:.1f}'.format(period, gamma))
        ax.set_ylabel('Cumulative return')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}\%'.format(y * 100)))
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        fig.subplots_adjust(left=0.1, bottom=0.23, right=0.78, top=0.92)
        fig_name = fig_file_name.format(key, int(gamma * 10))
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)
        savefig(fig_name)
        close()
        # show()
