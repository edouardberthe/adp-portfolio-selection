import numpy as np

from .parameters import beta, init


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

