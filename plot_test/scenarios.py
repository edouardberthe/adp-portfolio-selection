import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import cauchy, gaussian_kde, levy, norm, t

from data import Data, Returns, figsize
from generator import generateGaussianScenarios, generateStudentTScenarios


def test_generation_student(stock1, stock2, nu=4):
    r1 = Returns.iloc[:, stock1]
    r2 = Returns.iloc[:, stock2]
    bins1 = np.linspace(- r1.quantile(0.05), r1.quantile(0.95), 100)
    bins2 = np.linspace(- r2.quantile(0.05), r2.quantile(0.95), 100)
    plt.figure(1, figsize=(14, 8))
    plt.subplot(321)
    plt.hist(r1, normed=True, bins=bins1)
    plt.subplot(322)
    plt.hist(r2, normed=True, bins=bins2)
    students = generateStudentTScenarios(nu, 100000)[0]
    plt.subplot(323)
    plt.hist(students[:, stock1], normed=True, bins=bins1)
    plt.subplot(324)
    plt.hist(students[:, stock2], normed=True, bins=bins2)
    plt.subplot(325)
    gaussians = generateGaussianScenarios(100000)[0]
    plt.hist(gaussians[:, stock1], normed=True, bins=bins1)
    plt.subplot(326)
    plt.hist(gaussians[:, stock2], normed=True, bins=bins2)
    plt.show()


def comparison_densities(stock=0):
    """
    Plots the histogram of the stock's returns and different densities fitted to
    the data.
    """
    r = Returns.iloc[:, stock]
    bins = np.linspace(- 3 * r.std(), 3 * r.std(), 100)
    plt.figure(figsize=figsize)
    ax = plt.gca()
    r.hist(normed=True, bins=bins, label='{:s} returns'.format(Data.columns[stock]), alpha=0.5, ax=ax)
    rvs = [
        ('T 0.01', 'orange', t(df=0.01, loc=r.mean(), scale=r.std())),
        ('T 0.1', 'orange', t(df=0.1, loc=r.mean(), scale=r.std())),
        ('Gaussian', 'purple', norm(loc=r.mean(), scale=r.std())),
        ('Cauchy', 'green', cauchy(loc=r.mean(), scale=r.std())),
        (u'Lévy', 'yellow', levy(loc=r.mean(), scale=r.std())),
        ('Non Parametric', 'red', gaussian_kde(r))
    ]
    for (name, color, rv) in rvs:
        ax.plot(bins, rv.pdf(bins), label=name, lw=2, color=color)
    plt.legend()
    plt.show()


def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys
