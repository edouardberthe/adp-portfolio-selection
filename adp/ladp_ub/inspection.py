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

    # Indexer used to find index on a mask
    indexer = np.arange(N)

    # Step 0

    # Step 1

    # Selling Sell assets
    I = np.all((l > 0,  h[1:] > 0),  axis=0)  # Sell assets
    if I.any():
        y[I] = h[1:][I]
        h[1:][I] = 0

    # Selling Force-to-Sell assets
    F = np.all((l <= 0, h[1:] > w0), axis=0)  # Force-to-sell assets
    if F.any():
        y[F] = h[1:][F] - w0
        h[1:][F] = w0
    # Updating cash
    h[0] += (1-theta) * y.sum()

    # Step 2: if there is a Best asset, buying with cash
    idx = np.all((k > 0, h[1:] < w0), axis=0)     # Buy assets
    if idx.any():
        j_star = indexer[idx][np.argmax(k[idx])]  # Best buy
        while True:
            if w0 - h[j_star] < h[0] / (1+theta):
                x[j_star] = w0 - h[j_star+1]
                h[0] -= (1+theta) * x[j_star]
                h[j_star+1] = w0
                idx = np.all((k > 0, h[1:] < w0), axis=0)
                if idx.any():
                    j_star = indexer[idx][np.argmax(k[idx])]
                else:
                    break
            else:
                x[j_star] = h[0] / (1+theta)
                h[j_star+1] += h[0] / (1+theta) # useless but anyway
                h[0] = 0
                break

    return x, y


class LADPUBInspectionModel(LADPInspectionModel):
    solver = LADPUB
