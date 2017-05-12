import numpy as np
import numpy.random as rd

from adp.parameters import r
from data import Data


class GaussianGenerator:

    def __init__(self, freq='W-FRI'):
        self.Data = Data.asfreq(freq, method='pad')
        self.Gross = (self.Data / self.Data.shift())[1:]
        self.LogGross = np.log(self.Gross)
        self.mean = self.LogGross.mean()
        self.cov = self.LogGross.cov()

    def generate(self):
        LogScenarios = rd.multivariate_normal(self.mean, self.cov)
        return np.concatenate(((1+r,), np.exp(LogScenarios)))
