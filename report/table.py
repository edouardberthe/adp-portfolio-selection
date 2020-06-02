from time import time

from matplotlib import pyplot as plt
from tabulate import tabulate

from data import *
from entities.portfolio import EquallyWeightedPortfolio, PortfolioGroup
from generator import generate_gaussian, generate_t
from markowitz import Markowitz
from scenarios_based.models import CVaR, GMD, MAD, Minimax, ScenariosBasedPortfolioModel, SemiMAD

s, p = generate_gaussian(100)


def TestThree():
    MADport = MAD(s, p)
    SemiMADport = SemiMAD(s, p)
    GMDport = GMD(s, p)

    g = PortfolioGroup([MADport, SemiMADport, GMDport])
    g.plot()
    plt.show()

    MADport = MAD(s, p, output=True, Nmax=5)  
    SemiMADport = SemiMAD(s, p, output=True, Nmax=5)
    GMDport = GMD(s, p, output=True, Nmax=5)

    g = PortfolioGroup([MADport, SemiMADport, GMDport])
    g.plot()
    plt.show()


def TestWmin():
    MADport = MAD(s, p, output=True, Nmax=5)
    MADportWmin = MAD(s, p, output=True, Nmax=5, Wmin=0.1)
    GMDport = GMD(s, p, output=True, Nmax=5)
    GMDportWmin = GMD(s, p, output=True, Nmax=5, Wmin=0.1)
    g = PortfolioGroup([MADport, MADportWmin, GMDport, GMDportWmin])
    g.plot()
    plt.show()


def PlotRollingVolatility():
    ax = plt.figure(figsize=figsize).gca()
    Returns.rolling(365).std().dropna().plot(legend=False, title='CAC40 Rolling Volatilities', ax=ax)
    ax.set_ylabel('Volatility')
    plt.show()


def TestMADSemiMAD():
    """Checks that MAD and Semi MAD model lead to the same portfolio, and that MAD = 2 * SemiMAD."""
    s, p = generate_gaussian()
    MADPort = MAD(s, p).optimize().getPortfolio()
    SemiMADPort = SemiMAD(s, p).optimize().getPortfolio()
    PortfolioGroup([MADPort, SemiMADPort]).plot()


def TableInfluenceIntegerVariables():
    """
    Illustrates the influence of the additional Integer constraints (Nmax and Wmin) on computation time. Computes,
    draws and plots MAD Model computation time on N scenarios with N increasing, with the different parameters.
    """
    rows = []
    N = [10, 50, 100, 200, 300, 400, 500, 1000]
    params = [{'Wmin': Wmin, 'Nmax': Nmax} for Nmax in (None, 5) for Wmin in (None, 0.005)]
    print(params)
    for n in N:
        print("Computing with {:d} scenarios".format(n))
        s, p = generate_gaussian(n)
        row = [n]
        for param in params:
            t = time()
            MAD(s, p, **param).optimize()
            row.append(time() - t)
        rows.append(row)
    rows = np.array(rows)
    print("\nComputation times:\n")
    headers = ['N', 'LP', 'Wmin', 'Nmax', 'Wmin and Nmax']
    print(tabulate(rows, headers=headers))
    for i in range(1, 5):
        plt.plot(N, rows[:, i], label=headers[i])
    plt.legend(loc='upper left')
    plt.xlabel('Number of scenarios')
    plt.ylabel('Computation time (s)')
    plt.title('Influence of additional integer variables on computation time')
    plt.show()


def TableTime(N=5000, latex=False, generator=generate_t, **kwargs):
    """
    Computes and plots a Table comparing the different risk measures and safety measures. Important: this is NOT a good
    plot_test, because we suppose holding a portfolio from the beginning of times, but this portfolio is computed from the
    future values.
    :type N:      int  - Number of scenarios to generate.
    :type latex:  bool - If table output in LaTeX format (useful for the report / presentation)
    :type kwargs: expression - All arguments will go to the PortfolioOptimizer constructor, e.g 'output=True', or 'Nmax=5'.
    """
    models = [SemiMAD, Minimax, CVaR, Markowitz]
    s, p = generator(N)

    # First we add the Equally Weighted Portfolio case (for sake of comparison).
    eq = EquallyWeightedPortfolio()
    rows = [['Equally Weighted', eq.MeanReturn() * 100, eq.Return() * 100, eq.Vol() * 100, eq.MaxDrawDown() * 100, None]]
    for model in models:
        t = time()
        if issubclass(model, ScenariosBasedPortfolioModel):
            m = model(s, p, **kwargs)
        else:
            m = model(**kwargs)
        print("Computing", m.modelName)
        port = m.optimize().getPortfolio()
        rows.append([m.modelName, port.MeanReturn() * 100, port.Return() * 100, port.Vol() * 100,
                     port.MaxDrawDown() * 100, time() - t])
    print("\nComputating Table with {:d} scenarios\n".format(N))
    print(tabulate(rows, headers=['Model', 'Mean return (%)', 'Mean Cumulative Return (%)', 'Volatility (%)',
                                  'Max Draw Down(%)', 'Time (s)'], tablefmt=None if not latex else 'latex'))

if __name__ == '__main__':
    TableTime()
