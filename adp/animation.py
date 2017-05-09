from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from numpy import arange, ones, zeros

from adp.parameters import S, T
from data import Data, N


class ValueFunctionAnimation(FuncAnimation):

    def __init__(self, updater):
        self.updater = updater

        # Animated Plotting figure
        fig = plt.figure(figsize=(20, 10))
        self.ax = Axes3D(fig)
        # ax.set_ylim(-1.1, 1.1)
        # ax.set_xlim(0, 5)
        self.ax.set_zlim(0, 1.5)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Asset')
        self.ax.set_zlabel('Value function')
        self.ax.set_title('Linear Approximate Value Function')

        self.xs = arange(T+1)
        self.ys = ones(T+1)

        colors = 'bgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcm'
        self.lines = []
        self.lines.append(self.ax.plot(self.xs, 0 * self.ys, zeros(T+1), label='Cash', color=colors[0])[0])
        for i in range(1, N):
            self.lines.append(self.ax.plot(self.xs, i * self.ys, zeros(T+1), label=Data.columns[i-1],
                                           color=colors[i])[0])
        self.ax.legend()
        super().__init__(fig, self.run, S, blit=True, repeat=False)

    def run(self, s):
        next(self.updater)
        for (i, line) in enumerate(self.lines):
            line.set_data(self.xs, i * self.ys)
            line.set_3d_properties(self.updater.V[:, i])
        self.ax.figure.canvas.draw()
        return self.lines


def plot(V, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(20, 10))
    ax = Axes3D(fig)
    # ax.set_ylim(-1.1, 1.1)
    # ax.set_xlim(0, 5)
    ax.set_zlim(0, 1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Asset')
    ax.set_zlabel('Value function')
    ax.set_title('Linear Approximate Value Function')

    xs = arange(T+1)
    ys = ones(T+1)

    colors = 'bgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcm'
    lines = []
    lines.append(ax.plot(xs, 0 * ys, V[:,0], label='Cash', color=colors[0])[0])
    for i in range(1, N):
        lines.append(ax.plot(xs, i * ys, V[:,i], label=Data.columns[i - 1],
                                       color=colors[i])[0])
    ax.legend()
    return ax
