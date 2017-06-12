import numpy as np
from numpy import identity, zeros

from data import N
from parameters import beta, init


def CVaR(h):
    """
    Return the beta-CVaR associated to the terminal states h[0], h[1], ...
    :type h: np.ndarray
    """
    f = h.sum(axis=1) - init
    f.sort()
    S = len(f)
    l = int(np.ceil(S * (1-beta)))
    return - (f[:l-1].sum() / (S * (1-beta)) + f[l-1] * (1 - (l-1) / (S * (1 - beta))))

e = identity(N+1)


def generateΔCVaR(RT, h_plus):
    """
    Return the ΔCVaR associated to the states h_plus (at T-1) and the final returns RT.
    :param RT:      np.ndarray
    :param h_plus:  np.ndarray
    :return:        np.ndarray
    """
    CVaR_plus = zeros(N+1)
    for i in range(N+1):
        CVaR_plus[i] = CVaR(RT * (h_plus + [e[i]]))
    ΔCVaR = CVaR_plus - CVaR(RT * h_plus)
    return ΔCVaR