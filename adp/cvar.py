import numpy as np

from .parameters import beta


def CVaR(h):
    """
    Return the beta-CVaR associated to the terminal states h[0], h[1], ...
    :type h: np.ndarray
    """
    f = h.sum(axis=1)
    f.sort()
    l = np.ceil(len(h) * (1-beta))
    return (f[:l-1].sum() - (l-1) * f[l-1]) / (len(h) * (1-beta)) + f[l-1]
