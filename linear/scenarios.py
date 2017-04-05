# coding: utf8

from math import gamma, pi

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la, random as rd
from scipy import stats

from data import Data, MeanReturns, Returns, figsize

def generateGaussianScenarios(NbScenarios=1000, start=None, end=None, seed=None):
    """
    Generates random scenarios based on a multivariate Gaussian distribution of the log returns.
        - NbScenarios: int    - Number of scenarios to compute
        - start/end:   period - Period on which computing the variance-covariance matrix
        - seed:        int    - Seed for random generation
    """
    LocalData = Data[start:end]
    LogReturns = np.log(LocalData / LocalData.shift())[1:]
    MeanLogReturns = LogReturns.mean()
    CovLogReturns = LogReturns.cov()

    if seed is not None:
        rd.seed(seed)

    LogScenarios = rd.multivariate_normal(MeanLogReturns, CovLogReturns, size=NbScenarios)
    Scenarios = np.exp(LogScenarios) - 1

    Probas = np.ones(NbScenarios) / NbScenarios
    return Scenarios, Probas


def generateStudentTScenarios(NbScenarios=1000, nu=3, start=None, end=None, seed=None):
    """
    Generates random scenarios based on a multivariate 'student' t distribution of the log returns.
        - NbScenarios: int    - Number of scenarios to compute
        - start/end:   period - Period on which to compute the
                                variance-covariance matrix
        - seed:        int    - Seed for random generation
    """
    LocalData = Data[start:end]
    LocalReturns = (LocalData / LocalData.shift() - 1)[1:]
    MeanLocalReturns = LocalReturns.mean()
    CovLocalReturns = LocalReturns.cov()

    if seed is not None:
        rd.seed(seed)

    gaussian = rd.multivariate_normal(np.zeros(len(Data.columns)), CovLocalReturns, NbScenarios)
    chi2 = rd.chisquare(nu, (NbScenarios, 1))
    scenarios = gaussian / np.sqrt(nu / chi2) + np.array(MeanLocalReturns)
    probas = np.ones(NbScenarios) / NbScenarios
    return scenarios, probas


def testGenerationStudent(stock1, stock2, nu=4):
    r1 = Returns.iloc[:, stock1]
    r2 = Returns.iloc[:, stock2]
    bins1 = np.linspace(- r1.quantile(0.05), r1.quantile(0.95), 100)
    bins2 = np.linspace(- r2.quantile(0.05), r2.quantile(0.95), 100)
    fig = plt.figure(1, figsize=(14,8))
    fig.subplot(321)
    plt.hist(r1, normed=True, bins=bins1)
    plt.subplot(322)
    plt.hist(r2, normed=True, bins=bins2)
    students = generateStudentTScenarios(nu, 100000)
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


def ComparisonDensities(stock=0):
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
        ('T 0.01', 'orange', stats.t(df=0.01, loc=r.mean(), scale=r.std())),
        ('T 0.1', 'orange', stats.t(df=0.1, loc=r.mean(), scale=r.std())),
        ('Gaussian', 'purple', stats.norm(loc=r.mean(), scale=r.std())),
        ('Cauchy', 'green', stats.cauchy(loc=r.mean(), scale=r.std())),
        (u'Lévy', 'yellow', stats.levy(loc=r.mean(), scale=r.std())),
        ('Non Parametric', 'red', stats.gaussian_kde(r))
    ]
    for (name, color, rv) in rvs:
        ax.plot(bins, rv.pdf(bins), label=name, lw=2, color=color)
    plt.legend()
    plt.show()


def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def multivariate_t_pdf(x, mu, sigma, df):
    """
    Multivariate t-student density:
    output:
        the density of the given x element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
    """
    d = len(MeanReturns)
    Num = gamma(1. * (d + df) / 2)
    Denom = gamma(1. * df / 2) * pow(df * pi, 1. * d / 2) \
            * np.sqrt(la.det(sigma)) \
            * (1 + (1./df) * np.dot(np.dot((x - mu), la.inv(sigma)), (x - mu))) ** ((d+df)/2)
    d = 1. * Num / Denom 
    return d


def K(h):
    """Triangular Kernel."""
    def wrapped(x):
        return (np.abs(x) < 1./h) * h * (1 - h * np.abs(x))
    return wrapped


def kernel_density_estimator(x, h, data):
    return K(h)((x - Returns) / h).prod(axis=1).sum() / (len(data) * h.prod())
