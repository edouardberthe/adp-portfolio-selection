from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import ones

from adp.plot.base import PlotterProcess
from data import N
from parameters import T, figsize

colors = 'bgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykw'

class SlopesNumberPlotter(PlotterProcess):

    def __init__(self, V):
        super().__init__()
        self.V = V
        self.ax = Axes3D(self.fig)

    def draw(self):
        self.ax.clear()
        for n in range(N):
            xs = range(T)
            ys = n * ones(T)
            zs = [len(self.V[t].V[n].a) for t in range(T)]
            self.ax.plot(xs, ys, zs=zs, color=self.colors[n])
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Asset')
        self.ax.set_zlabel('Number of slopes')
        self.ax.set_title('Number of slopes in value function')


class MeanBreaksPlotter(PlotterProcess):

    def __init__(self, V):
        super().__init__()
        self.V = V
        self.ax = Axes3D(self.fig)

    def draw(self):
        self.ax.clear()
        for n in range(N):
            xs = range(T)
            ys = n * ones(T)
            zs = [Vt.V[n].a.mean() for Vt in self.V]
            self.ax.plot(xs, ys, zs=zs, color=self.colors[n])
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Asset')
        self.ax.set_zlabel('Mean of breaks')
        self.ax.set_title('Average break')


class MeanSlopesPlotter(PlotterProcess):

    def __init__(self, V):
        super().__init__()
        self.V = V
        self.ax = Axes3D(self.fig)

    def draw(self):
        self.ax.clear()
        for n in range(N):
            xs = range(T)
            ys = n * ones(T)
            zs = [Vt.V[n].slopes.mean() for Vt in self.V]
            self.ax.plot(xs, ys, zs=zs, color=self.colors[n])
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Asset')
        self.ax.set_zlabel('Mean of Slopes')
        self.ax.set_title('Average slope')


class FirstSlopeAx(Axes3D):

    def __init__(self, strategy):
        super().__init__(plt.figure(figsize=figsize))
        self.strategy = strategy

    def plot(self):
        super().clear()
        for n in range(N):
            xs = range(T)
            ys = n * ones(T)
            zs = [V[n].slopes[0] / V.cash for V in self.strategy]
            super().plot(xs, ys, zs=zs, color=colors[n])
        super().set_xlabel('Time')
        super().set_ylabel('Asset')
        super().set_zlabel('Slope')
        super().set_title('First slope')
