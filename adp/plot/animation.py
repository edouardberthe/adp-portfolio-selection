from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from numpy import arange, ones, zeros

from data import Data, N
from parameters import S, T, repeat


class ValueFunctionAnimation(FuncAnimation):

    def __init__(self, updater):
        self.updater = updater

        # Animated Plotting figure
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        # ax.set_ylim(-1.1, 1.1)
        # ax.set_xlim(0, 5)
        self.ax.set_zlim(0, 1.5)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Asset')
        self.ax.set_zlabel('Value function')
        self.ax.set_title('Linear Approximate Value Function')

        self.xs = arange(T)
        self.ys = ones(T)

        colors = 'bgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcm'
        self.lines = []
        self.lines.append(self.ax.plot(self.xs, 0 * self.ys, zeros(T),
                                       label='Cash', color=colors[0])[0])
        for i in range(1, N):
            self.lines.append(self.ax.plot(self.xs, i * self.ys, zeros(T),
                                           label=Data.columns[i-1],
                                           color=colors[i])[0])
        self.ax.legend()
        super().__init__(self.fig, self.run, S-1, blit=True, repeat=False)

    def run(self, s):
        for i in range(repeat):
            next(self.updater)
        for (i, line) in enumerate(self.lines):
            line.set_data(self.xs, i * self.ys)
            line.set_3d_properties(self.updater.V[:, i])
        self.ax.set_title("Scenario {:d}".format(s))
        self.fig.canvas.draw()
        return self.lines
