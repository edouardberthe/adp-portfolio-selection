import numpy as np
import numpy.random as rd
from rpy2.robjects import pandas2ri, r as R

from data import Data, N
from parameters import freq


class GaussianGenerator:

    def __init__(self, r, start=None, end=None):
        self.r = r
        self.Gross = (Data.asfreq(freq, method='pad').pct_change()+1)[1:][start:end]
        self.LogGross = np.log(self.Gross)
        self.mean = self.LogGross.mean()
        self.cov = self.LogGross.cov()

    def generate(self):
        LogScenarios = rd.multivariate_normal(self.mean, self.cov)
        return np.concatenate(((1 + self.r,), np.exp(LogScenarios)))


class OGARCHGenerator:

    def __init__(self, start=None, end=None):
        self.Gross = (Data.asfreq(freq, method='pad').pct_change()+1)[1:][start:end]
        self.LogGross = np.log(self.Gross)
        self.mean = self.LogGross.mean()
        self.cov = self.LogGross.cov()

    def generate(self):
        pandas2ri.activate()
        R.assign('Data', self.Data)
        R.assign('N', N)
        R("""
library(rmgarch)
xspec = ugarchspec(mean.model = list(armaOrder = c(1, 1)), variance.model = list(garchOrder = c(1,1), model = 'sGARCH'), distribution.model = 'norm')
uspec = multispec(replicate(N, xspec))
spec = dccspec(uspec = uspec, dccOrder = c(1, 1), distribution = 'mvnorm')
speca = dccspec(uspec = uspec, dccOrder = c(1, 1), model='aDCC', distribution = 'mvnorm')
fit_adcc = dccfit(spec1a, data = Data)
""")

