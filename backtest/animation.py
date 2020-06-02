from matplotlib import animation, pyplot as plt

from generator import TGenerator
from backtest.backtest import BackTest
from backtest.param_pool import BackTestParamPool
from backtest.group import BackTestGroup
from entities.model import EWPortfolioModel
from scenarios_based.models import CVaR, Minimax, SemiMAD


def animate_backtest(backtest: BackTest):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2, label=backtest.Model.__name__)
    plt.legend(loc='lower right')  # We hope our backtest won't go there!
    plt.title(str(backtest.pool))
    ax.grid()
    xdata = [backtest.index[0]]
    ydata = [0]

    def run(x):
        date, gross = x
        # Draw with the new portfolio
        xdata.append(date)
        ydata.append((ydata[-1] + 1) * gross - 1)
        line.set_data(xdata, ydata)

        ymin, ymax = ax.get_ylim()
        if ydata[-1] >= ymax * 0.8:
            ax.set_ylim(ymin, ymax * 1.5)
            ax.figure.canvas.draw()
        elif ydata[-1] <= ymin * 0.8:
            ax.set_ylim(ymin * 1.5, ymax)
            ax.figure.canvas.draw()

        return line,

    def init():
        ax.set_xlim(backtest.index[0], backtest.index[-1])
        ax.set_ylim(-1, 1)
        line.set_data(xdata, ydata)
        return line,

    ani = animation.FuncAnimation(fig, run, backtest.generator, blit=False, interval=10, repeat=False, init_func=init)
    plt.show()


def animate_backtest_group(group):
    fig, ax = plt.subplots()
    lines = [ax.plot([], [], lw=2, label=b.Model.__name__)[0] for b in group]
    plt.legend(loc='lower right')
    plt.title(str(group.pool))
    ax.grid()
    xdata = [group.index[0]]
    ydata = [[0] for b in group]

    def run(x):
        date, grosses = x
        # Draw with the new portfolio
        xdata.append(date)
        for i in range(len(group)):
            ydata[i].append((ydata[i][-1] + 1) * grosses[i] - 1)
            lines[i].set_data(xdata, ydata[i])

        ymin, ymax = ax.get_ylim()
        if max(ydata[-1]) >= ymax * 0.8:
            ax.set_ylim(ymin, ymax * 1.5)
            ax.figure.canvas.draw()
        elif min(ydata[-1]) <= ymin * 0.8:
            ax.set_ylim(ymin * 1.5, ymax)
            ax.figure.canvas.draw()

        return lines,

    def init():
        ax.set_xlim(group.index[0], group.index[-1])
        ax.set_ylim(-1, 1)
        for i in range(len(group)):
            lines[i].set_data(xdata, ydata[i])
        return lines,

    ani = animation.FuncAnimation(fig, run, group.generator, blit=False, interval=10, repeat=False, init_func=init)
    plt.show()


if __name__ == '__main__':
    generator = TGenerator(3)
    pool = BackTestParamPool(freq='M', window=365, generator=generator, N=1000)
    # b = BackTest(SemiMAD, pool)
    # animateBackTest(b)
    g = BackTestGroup([CVaR, Minimax, SemiMAD, EWPortfolioModel], pool)
    animate_backtest_group(g)
