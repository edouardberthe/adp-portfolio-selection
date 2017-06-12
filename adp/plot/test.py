import numpy as np

from adp.plot.base import PlotterProcess
from data import Data
from parameters import test_end, test_start, init


class GrossTestPlotter(PlotterProcess):

    def __init__(self, trainer, lengths=(10, 50)):
        super().__init__()
        self.trainer = trainer
        self.ax = self.fig.gca()
        self.scenarios = []
        self.test_wealth = []
        self.gross = (Data.asfreq('W-FRI', method='pad').pct_change()+1)[1:][test_start:test_end]
        self.lengths = lengths

    def draw(self):
        self.scenarios.append(self.trainer.counter)
        self.test_wealth.append(np.sum(self.trainer.strategy.score(self.gross)) / init)

        self.ax.clear()
        # We plot the line 1
        self.ax.plot([0, self.scenarios[-1]], [1, 1])
        # We plot the returns
        self.ax.plot(self.scenarios, self.test_wealth)
        for l in self.lengths:
            self.ax.plot(self.scenarios, [np.mean(self.test_wealth[max(i+1-l, 0):i+1]) for i in range(len(self.test_wealth))])
        # Config
        self.ax.set_xlabel('Scenarios')
        self.ax.set_ylabel('Gross')
        self.ax.set_title('Test Gross Return')
