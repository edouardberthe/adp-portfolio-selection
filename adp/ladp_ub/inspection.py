import numpy as np
from numpy import zeros

from adp.ladp.inspection import LADPInspectionModel
from adp.parameters import theta, w0
from data import N


def LADPUB(obj, R: np.ndarray, h_plus: np.ndarray, u: np.array):
    """
    Returns the Linear Approximated Dynamic Programming solution by inspection.
    :param R     : Return at t
    :param h_plus: h^+_(t-1), state variable
    :param u     : u_it, old value function slopes
    :rtype       : tuple(np.array, np.array, np.array)
    """

    # Computing pre-decision variables
    h = h_plus * R

    # Return values
    x = zeros(N)
    y = zeros(N)

    k =   u[1:] - (1 + theta) * u[0]          # Buying slopes
    l = - u[1:] + (1 - theta) * u[0]          # Selling slopes

    # Step 0
    I = np.all((l > 0,  h[1:] > 0),  axis=0)  # Sell assets
    F = np.all((l <= 0, h[1:] > w0), axis=0)  # Force-to-sell assets
    indexer = np.arange(N)
    idx = np.all((k > 0, h[1:] < w0), axis=0)
    j_star = indexer[idx][np.argmax(k[idx])] if idx.any() else None  # Best buy

    # Step 1
    if I.any():
        y[I] = h[1:][I]

    if F.any():
        y[F] = h[1:][F] - w0

    h[0] += (1-theta) * (h[1:][I].sum() + h[1:][F].sum())

    # Step 2
    while j_star is not None:
        if w0 - h[j_star] < h[0] / (1+theta):
            x[j_star] = w0 - h[j_star+1]
            h[0] -= (1+theta) * x[j_star]
            h[j_star+1] = w0
            idx = np.all((k > 0, h[1:] < w0))
            if idx.any():
                j_star = indexer[idx][np.argmax(k[idx])]
            else:
                break
        else:
            x[j_star] = h[0]
            h[j_star+1] += h[0] / (1+theta)
            h[0] = 0
            break

    return x, y


class LADPUBInspectionModel(LADPInspectionModel):
    solver = LADPUB
