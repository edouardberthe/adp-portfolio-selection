from mpl_toolkits.mplot3d import Axes3D
from numpy import ones, zeros

from adp.plot.base import PlotterProcess
from adp.strategy import ADPStrategy
from data import Data
from parameters import T


class PWLValueFunctionPlotter(PlotterProcess):

    def __init__(self, n: int, strategy: ADPStrategy):
        super().__init__()
        self.strategy = strategy
        self.n = n
        self.ax = Axes3D(self.fig)
        self.ax.set_title(Data.columns[self.n])

    def draw(self):
        width = 0.1
        colors = 'bgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykw'
        for t in range(T):
            V = self.strategy[t][self.n]
            x = V.x()
            xs = x[:-1]
            ys = t * ones(len(xs))
            dx = x[1:] - x[:-1]
            dy = zeros(len(xs)) + width
            bot = zeros(len(xs))
            top = V.slopes
            self.ax.bar3d(xs, ys, bot, dx, dy, top, color=colors[t])
        self.ax.set_xlabel('Position')
        self.ax.set_ylabel('Time')
        self.ax.set_zlabel('Slope of Value function')
