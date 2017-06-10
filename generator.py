from math import gamma as Gamma, pi

import numpy as np
from numpy import linalg as la, random as rd

from data import Data, MeanReturns


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
    Num = Gamma(1. * (d + df) / 2)
    Denom = Gamma(1. * df / 2) * pow(df * pi, 1. * d / 2) \
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
