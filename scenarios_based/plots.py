from time import time

from matplotlib import pyplot as plt

from data import Returns, figsize
from entities import PortfolioGroup
from .models import CVaR, MAD, Minimax, SemiMAD
from .scenarios import generateGaussianScenarios, generateStudentTScenarios


def plotCompareMADSemiMADOutput():
    """Checks (and plots) that SemiMad = MAD / 2 and both methods give the same portfolio."""
    s, p = generateGaussianScenarios(1000)
    PortfolioGroup([MAD(s, p).optimize().getPortfolio(), SemiMAD(s, p).optimize().getPortfolio()]).plot()


def plotCompareMADSemiMADPerformance():
    Nrange = range(500, 5001, 500)
    MADtimes = []
    SemiMADtimes = []
    for n in Nrange:
        print(n)
        s, p = generateGaussianScenarios(n)
        t = time()
        MAD(s, p).optimize()
        MADtimes.append(time() - t)
        t = time()
        SemiMAD(s, p).optimize()
        SemiMADtimes.append(time() - t)
    plt.figure(figsize=figsize)
    plt.plot(Nrange, MADtimes, label='MAD')
    plt.plot(Nrange, SemiMADtimes, label='Semi-MAD')
    plt.title('Computation time comparison between MAD and Semi-MAD')
    plt.xlabel('Number of scenarios generated')
    plt.ylabel('Computation time')
    plt.legend()
    plt.show()


def plotCompareScenariosGenerators(nu=3):
    """
    :param nu: float
    """
    seed = 4
    g, _ = generateGaussianScenarios(1000, seed=seed)
    t, _ = generateStudentTScenarios(1000, nu=nu, seed=seed)

    plt.figure(figsize=figsize)
    plt.plot(Returns.iloc[:, 0], Returns.iloc[:, 1], 'o', color='green', alpha=0.5, label='Data')
    plt.plot(g[:, 0], g[:, 1], 'o', color='blue', label='Gaussian')
    plt.plot(t[:, 0], t[:, 1], 'o', color='red', label='Student')
    plt.axis((-0.15, 0.2, -0.2, 0.3))
    plt.xlabel('{:s} returns'.format(Returns.columns[0]))
    plt.ylabel('{:s} returns'.format(Returns.columns[1]))
    plt.legend()
    plt.show()


def timeReconfigure(model, N, M, reconfigure):
    """
    Computes the time to optimizing M times the model with N scenarios with (or without) reconfiguration, then
    divides the total time by M to have the average time per optimization.
    :type model:      scenarios_based.models.model.ScenariosBasedPortfolioModel - model to optimize
    :type N:          int                                      - Number of scenarios to generate
    :type M:          int                                      - Number of optimizations to make
    :type reconfigure bool                                     - Do we reconfigure or recreate the model each time?
    """
    s, p = generateGaussianScenarios(N)
    t = time()
    if reconfigure:
        m = model(s, p).update()
    for i in range(M):
        s, p = generateGaussianScenarios(N)
        if reconfigure:
            m.reconfigure(s, p).optimize()
        else:
            model(s, p).optimize()
    return (time() - t) / M


def plotReconfigureComparison():
    """Compares the advantage of reconfiguring versus not reconfiguring."""
    models = [MAD, SemiMAD, Minimax, CVaR]
    colors = ['red', 'green', 'yellow', 'blue', 'orange']
    N = range(100, 1000, 100)
    M = 10
    plt.figure(figsize=figsize)
    for i, model in enumerate(models):
        print(model.__name__)
        for reconfigure in [True, False]:
            print("\t", "With reconfiguration" if reconfigure else "Without reconfiguration")
            times = []
            for n in N:
                print("\t\t", n, "scenarios")
                times.append(timeReconfigure(model, n, M, reconfigure))
            if reconfigure:
                plt.plot(N, times, '--', color=colors[i])
            else:
                plt.plot(N, times, '-', color=colors[i], label=model.__name__)
    plt.legend(loc='upper left')
    plt.ylabel('Time / optimization (s)')
    plt.xlabel('Number of scenarios')
    plt.show()



if __name__ == '__main__':
    s, p = generateGaussianScenarios(100)
    plotCompareMADSemiMADOutput()
    plotCompareMADSemiMADPerformance()
    #plotReconfigure(GMD, N=100)
