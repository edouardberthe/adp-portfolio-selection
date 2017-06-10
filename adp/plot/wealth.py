from adp.plot.base import PlotterProcess
from parameters import init


class FinalReturnPlotter(PlotterProcess):

    def __init__(self, trainer, lengths):
        super().__init__()
        self.trainer = trainer
        self.lengths = lengths
        self.ax = self.fig.gca()

    def draw(self):
        ret = self.trainer.memory.h / init
        self.ax.clear()
        self.ax.plot(ret)
        self.ax.plot([0, len(ret)], [1, 1])
        for l in self.lengths:
            self.ax.plot([ret[max(i+1-l, 0):i+1].mean() for i in range(len(ret))])
        self.ax.set_ylim(0.8, 1.2)