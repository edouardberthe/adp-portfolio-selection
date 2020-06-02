from abc import ABC, abstractmethod
from typing import Tuple, Sequence

import numpy as np
from numpy import random as rd

from data import Data, Returns


class ScenarioGenerator(ABC):
    def __init__(self, seed: int = None):
        if seed is not None:
            rd.seed(seed)

    @abstractmethod
    def generate(self, n_scenarios: int, start=None, end=None) -> Tuple[Sequence, Sequence]:
        raise NotImplementedError()

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()


class GaussianGenerator(ScenarioGenerator):
    def __str__(self) -> str:
        return "GaussianGenerator"

    def generate(self, n_scenarios: int, start=None, end=None) -> Tuple[Sequence, Sequence]:
        """
        Generates random scenarios based on a multivariate Gaussian distribution of the log returns.
            - NbScenarios: int    - Number of scenarios to compute
            - start/end:   period - Period on which computing the variance-covariance matrix
            - seed:        int    - Seed for random generation
        """
        data = Data[start:end]
        log_returns = np.log(data / data.shift())[1:]
        log_scenarios = rd.multivariate_normal(log_returns.mean(), log_returns.cov(), size=n_scenarios)
        scenarios = np.exp(log_scenarios) - 1
        probas = np.ones(n_scenarios) / n_scenarios
        return scenarios, probas


class TGenerator(ScenarioGenerator):

    def __init__(self, nu: float, seed: float = None):
        super().__init__(seed)
        self.nu = nu

    def __str__(self) -> str:
        return f"TGenerator({self.nu})"

    def generate(self, n_scenarios: int, start=None, end=None) -> Tuple[Sequence, Sequence]:
        """
        Generates random scenarios based on a multivariate 'student' t distribution of the log returns.
            - n_scenarios: int    - Number of scenarios to compute
            - start/end:   period - Period on which to compute the
                                    variance-covariance matrix
            - seed:        int    - Seed for random generation
        """
        data = Data[start:end]
        returns = data.pct_change()[1:]

        gaussian = rd.multivariate_normal(np.zeros(len(Data.columns)), returns.cov(), n_scenarios)
        chi2 = rd.chisquare(self.nu, (n_scenarios, 1))
        scenarios = gaussian / np.sqrt(self.nu / chi2) + np.array(returns.mean())
        probas = np.ones(n_scenarios) / n_scenarios
        return scenarios, probas


def K(h):
    """Triangular Kernel."""

    def wrapped(x):
        return (np.abs(x) < 1. / h) * h * (1 - h * np.abs(x))

    return wrapped


def kernel_density_estimator(x, h, data):
    return K(h)((x - Returns) / h).prod(axis=1).sum() / (len(data) * h.prod())
