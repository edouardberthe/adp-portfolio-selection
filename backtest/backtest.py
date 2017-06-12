from gurobipy import GurobiError

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data import Data, figsize
from entities.portfolio import EquallyWeightedPortfolio
from entities.model import EWPortfolioModel
from scenarios_based.models import SemiMAD
from generator import generateStudentTScenarios


class BackTestParamPool(object):
    """Represents a set of parameters for a BackTest."""

    def __init__(self, freq, window, generator, N, reconfigure=True):
        """
        :type freq: string | pandas.DateOffset - Frequency of rebalancing, with anchor for exact rebalancing date, ex:
                                                    - weekly: 'W-MON', 'W-TUE', etc.
                                                    - fornightly: '2W-MON', '2W-FRI', etc.
                                                    - monthly: 'M' (rebalacing last day of the month)
                                                    - annually: 'A-JAN', 'A-DEC', etc.
        :type window:      int                 - Time window (in days) to compute the rolling mean / covariance matrix
        :type generator:   function            - Scenarios generator
        :type N:           int                 - Number of scenarios to generate at each rebalancing date
        :type reconfigure: bool                - Do we have to use the reconfigure method in the Models?
        """
        self.freq = freq
        self.window = pd.Timedelta(days=window)
        self.generator = generator
        self.N = N
        self.reconfigure = reconfigure

    def __str__(self):
        return "Params: {:d} scenarios - {:s} rebalancing - Rolling window: {:d} days - Generator: {:s}".format(
            self.N,
            self.freq,
            self.window.days,
            "Gaussian" if self.generator.__name__ == 'generateGaussianScenarios' else "T"
        )

    def __repr__(self):
        return "<BackTest Parameters Pool {:s}".format(self)


class BackTest(object):
    """Simulates a portfolio rebalanced at each rebalancing date."""

    def __init__(self, Model, pool, **params):
        """
        Computes a BackTest of the model, by simulating a rebalancing portfolio.
        :type Model:  entities.PortfolioOptimizer - Model to BackTest (MAD, GMD, ...)
        :type pool:   BackTestParamPool       - Parameters of this BackTest
        :type params: dict                    - Parameters will be passed to each Model generated (Wmin, Nmax, ...)
        """
        self.Model = Model
        self.pool = pool
        self.PeriodicData = Data.asfreq(pool.freq, method='pad')
        self.PeriodicGrossReturns = (self.PeriodicData.shift(-1) / self.PeriodicData)[:-1]

        # We store the index on which we will do the computation (we need at least 'window' time of previous data to
        # generate the data).
        self.index = self.PeriodicGrossReturns[self.PeriodicGrossReturns.index[0] + pool.window:].index

        self.params = params
        self.portfolios = pd.DataFrame(columns=Data.columns)

    def generator(self):
        """This function is a python generator: for each rebalancing date, it yields the new portfolio."""
        if self.pool.reconfigure:
            s, p = self.pool.generator(self.pool.N)
            model = self.Model(s, p, name='BackTest', **self.params)
            model.update()
        for date in self.index:
            s, p = self.pool.generator(self.pool.N, nu=4, start=date - self.pool.window, end=date)
            if self.pool.reconfigure:
                model.reconfigure(s, p)
            else:
                model = self.Model(s, p, name='BackTest', **self.params)
            model.optimize()
            try:
                port = model.getPortfolio()
                print(date, model.objval)
            except GurobiError:
                print(date, "model infeasible, E.W. portfolio generated")
                port = EquallyWeightedPortfolio()
            self.portfolios.loc[date] = port
            yield date, (port * self.PeriodicGrossReturns.loc[date]).sum()

    def compute(self):
        generator = self.generator()
        try:
            while next(generator):
                pass
        except StopIteration:
            pass

    def ComputedPeriodicGrossReturns(self):
        return (self.portfolios * self.PeriodicGrossReturns).dropna().sum(axis=1)

    def plot(self, ax=None):
        if ax is None:
            ax = plt.figure(figsize=figsize).gca()
            plt.legend(loc='upper left')
            plt.xlabel('Time')
            plt.ylabel('Cumulative return')
            plt.title("RRR={:.0%}".format(self.params['RRR']) if 'RRR' in self.params else '')

        CumReturns = self.ComputedPeriodicGrossReturns().cumprod() - 1
        ":type: pandas.DataFrame"
        CumReturns.plot(label='{:s}, $\mu$={:.1%}, $\sigma$={:.1%}'.format(self.Model.__name__, self.annualizedReturn(),
                self.annualizedVol()), ax=ax)
        return ax

    def years(self):
        """
        Returns the number of years (float) on which is computed the backtest.
        :rtype float
        """
        return (self.index[-1] - self.index[0]).days / 365.25

    def annualizedReturn(self):
        return self.ComputedPeriodicGrossReturns().prod() ** (1 / self.years()) - 1

    def annualizedVol(self):
        return self.ComputedPeriodicGrossReturns().std() * np.sqrt(len(self.index) / self.years())


class BackTestGroup(list):
    """
    Represents a set of Backtests with the same parameters. The goal is to compute different samples of the same
    risk/safety measures or to compare different risk measures and safety measures.
    """

    def __init__(self, models, pool):
        """
        :type models: list              - List of PortfolioOptimizer to backtest
        :type pool:   BackTestParamPool - Parameters to apply to all models
        """
        super().__init__([BackTest(model, pool) for model in models])
        self.pool = pool
        self.PeriodicData = Data.asfreq(pool.freq, method='pad')
        self.index = self.PeriodicData[self.PeriodicData.index[0] + pool.window:].index[:-1]

    def generator(self):
        generators = [backtest.generator() for backtest in self]
        for date in self.index:
            res = []
            for generator in generators:
                res.append(generator.next()[1])
            yield date, res

    def compute(self):
        for backtest in self:
            backtest.compute()

    def plot(self):
        ax = plt.figure(figsize=figsize).gca()
        for backtest in self:
            backtest.plot(ax=ax)
        plt.title(str(self.pool))
        plt.legend(loc='upper left')
        plt.show()

if __name__ == '__main__':
    pool = BackTestParamPool(freq='M', window=365, generator=generateStudentTScenarios, N=1000)
    g = BackTestGroup([SemiMAD, SemiMAD, SemiMAD, EWPortfolioModel], pool)
    g.compute()
    g.plot()
