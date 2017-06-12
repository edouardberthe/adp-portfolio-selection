import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from numpy import random as rd

from data import A, Data, figsize, Gross, Returns


class Portfolio(pd.Series):

    def plot(self):
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        super().plot(kind='bar', rot=70, ax=ax)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        plt.subplots_adjust(bottom=0.15)
        plt.show()

    def DailyGross(self) -> pd.Series:
        """
        Computes the daily gross returns of this portfolio.
        :rtype: pandas.Series
        """
        return (Gross * self).sum(axis=1)

    def AnnualizedReturn(self):
        """
        Computes the annualized return of the portfolio, taking account of the compounding return (geometric mean).
        :rtype: float
        """
        return self.DailyGross().prod() ** (252. / len(Returns)) - 1

    def MeanReturn(self):
        """
        Computes the (arithmetic) mean return of the portfolio.
        :rtype: float
        """
        return (self * Returns.mean()).sum()

    def Vol(self):
        """
        :return The annualized volatility of the portfolio.
        :rtype float
        """
        return self.DailyGross().std() * np.sqrt(252)

    def MAD(self, scenarios, probas):
        """
        Computes the Mean Absolute Deviation of this portfolio with respect to the given scenarios / probas.
        :type scenarios: numpy.array
        :type probas:    numpy.array
        :rtype: float
        """
        Y = scenarios.dot(self)
        mu = probas.dot(scenarios).dot(self)
        D = np.abs(Y - mu)
        return probas.dot(D)
    
    def SemiMAD(self, scenarios, probas):
        """
        Computes the Semi Mean Absolute Deviation of this portfolio with respect to the given scenarios / probas.
        :rtype: float
        """
        Y = scenarios.dot(self)
        mu = probas.dot(scenarios).dot(self)
        D = (mu - Y) * (mu > Y)
        return probas.dot(D)

    def WorstReturn(self, scenarios):
        """
        Computes the worst return, given the possible scenarios.
        :rtype: float
        """
        return scenarios.dot(self).min()

    def MaxDrawDown(self):
        """Computes the maximal drawdown, which is the REAL worst daily return in the data using this portfolio."""
        return self.DailyGross().min() - 1

    def __call__(self, start, stop):
        """Computes the return of this portfolio on this slice of time."""
        data = Data[start:stop]
        return (data.loc[-1] / data.loc[0])


class RandomPortfolio(Portfolio):
    def __init__(self):
        port = rd.rand(len(A))
        super().__init__(data=port / port.sum(), index=Data.columns)


class EquallyWeightedPortfolio(Portfolio):
    def __init__(self):
        super().__init__(data=np.ones(len(A)) / len(A), index=Data.columns)


class PortfolioGroup(pd.DataFrame):

    def __init__(self, portfolios, **kwargs):
        super().__init__({p.name: p for p in portfolios}, **kwargs)

    def plot(self):
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        super().plot(kind='bar', rot=70, ax=ax)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        plt.subplots_adjust(bottom=0.15)
        plt.show()
