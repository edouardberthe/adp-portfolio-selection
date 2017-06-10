from mpl_toolkits.mplot3d import Axes3D
from numpy import ones

from adp.plot.base import PlotterProcess
from data import N


class FinalPositionsPlotter(PlotterProcess):

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.ax = Axes3D(self.fig)

    def draw(self):
        memory = self.trainer.memory
        self.ax.clear()
        for n in range(N):
            s = len(memory.RT)
            xs = range(s)
            ys = n * ones(s)
            zs = memory.hp[:, n+1]
            self.ax.plot(xs, ys, zs, color=self.colors[n])
        self.ax.set_xlabel('Scenarios')
        self.ax.set_ylabel('Asset')
        self.ax.set_zlabel('Position')
        self.ax.set_title('Final Portfolio composition')

