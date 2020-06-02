import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from backtest.group import BackTestGroup
from backtest.param_pool import BackTestParamPool
from data import Data, figsize
from entities.portfolio import EquallyWeightedPortfolio
from entities.model import EWPortfolioModel
from scenarios_based.models import SemiMAD
from generator import TGenerator


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
            except Exception:
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


if __name__ == '__main__':
    pool = BackTestParamPool(freq='M', window=365, generator=TGenerator(nu=3), N=1000)
    g = BackTestGroup([SemiMAD, SemiMAD, SemiMAD, EWPortfolioModel], pool)
    g.compute()
    g.plot()
