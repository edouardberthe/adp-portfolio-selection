from matplotlib import pyplot as plt

from backtest import BackTest


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