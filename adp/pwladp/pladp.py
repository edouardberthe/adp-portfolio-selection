import numpy as np
from numpy import ones, zeros, identity

from adp.ladp import LADPInspectionModel
from adp.transition import ft
from data import N
from .parameters import M, a, da, theta


def apply(u, u0, h_plus):
    h_mat = h_plus[1:].reshape(N, 1).dot(ones((1, M+1)))
    a_mat = ones((N, 1)).dot(a.reshape(1, M+1))
    idx_mat = a_mat < h_mat
    idx = idx_mat.argmin(axis=1)

    return u0 * h_plus[0] \
           + ((u[:, 1:][idx_mat[:, :-1]] - u[:, :-1][idx_mat[:, :-1]]) * a_mat[idx_mat]).sum() \
           + (u[range(N), idx] * h_plus[1:]).sum(), idx


def PLADP(R: np.ndarray, h_plus: np.ndarray, u: np.ndarray, u0: float):
    """
    Returns the Linear Approximated Dynamic Programming solution by inspection.
    :param R     : Return at t
    :param h_plus: h^+_(t-1), state variable
    :param u     : np.array old value function for the assets
    :param u0    : u_0t, old slope value function of the cash
    :rtype       : tuple(np.array, np.array, np.array)
    """
    # Computing pre-decision variables
    h = h_plus * R

    h_mat = h[1:].reshape(N, 1).dot(ones((1, M+1)))
    a_mat = ones((N, 1)).dot(a.reshape(1, M+1))
    idx_mat = a_mat < h_mat
    idx = np.argmin(idx_mat, axis=1) - 1

    # Computing approximate pre-decision variables
    h0 = h[0]
    h_tidle = zeros((N, M+1))
    h_tidle[idx_mat] = da

    for i in range(N):
        h_tidle[i, idx[i]] = h[1:][i] - a[idx[i]]

    # Return values
    x = zeros((N, M+1))
    y = zeros((N, M+1))

    k =   u - (1 + theta) * u0  # Buying slopes
    l = - u + (1 - theta) * u0  # Selling slopes

    # Step 0
    I = np.logical_and(l > 0, h_tidle > 0)            # Sell assets
    j_star_idx = np.logical_and(k > 0, h_tidle < da)
    j_star = np.unravel_index((k * j_star_idx).argmax()) if j_star_idx.any() else None  # Best buy

    # Step 1: selling Sell assets
    if I.any():
        y[I] = h_tidle[I]
        h0 += (1-theta) * y[I].sum()

    # Step 2
    while j_star is not None:
        if da - h_tidle[j_star] < h0 / (1+theta):
            x[j_star] = da - h_tidle[j_star]
            h0 -= (1+theta) * x[j_star]
            h[j_star] = da

            j_star_idx = np.logical_and(k > 0, h_tidle < da)
            if j_star_idx.any():
                j_star = np.unravel_index((k * j_star_idx).argmax(), k.shape)
            else:
                break
        else:
            x[j_star] = h0 / (1+theta)
            h[j_star] += x[j_star]      # ?
            h0 = 0
            break

    # Step 3: Perform sell-to-buy transactions
    if j_star is not None:

        i_star_idx = np.logical_and(l <= 0, h > 0, (1-theta)/(1+theta) * k[j_star] + l > 0)

        if i_star_idx.any():
            i_star = np.unravel_index((l * j_star_idx).argmax(), l.shape)

            while i_star is not None and j_star is not None:
                if h_tidle[i_star] < (1+theta)/(1-theta) * (da - h_tidle[j_star]):
                    y[i_star] += h[i_star]
                    x[j_star] += (1+theta)/(1-theta) * h[i_star]
                    h[i_star] -= y[i_star]
                    h[j_star] += x[j_star]

                    i_star_idx = np.logical_and(l <= 0, h > 0, (1-theta)/(1+theta) * k[j_star] + l > 0)
                    if i_star_idx.any():
                        i_star = np.unravel_index((l * i_star_idx).argmax(), l.shape)
                    else:
                        break

                else:
                    y[i_star] += (1+theta)/(1-theta) * (da - h_tidle[j_star])
                    x[i_star] += da - h_tidle[j_star]
                    h[i_star] -= y[i_star]
                    h[j_star] += x[j_star]

                    j_star_idx = np.logical_and(k > 0, h < da)
                    if j_star_idx.any():
                        j_star = np.unravel_index((k * j_star_idx).argmax(), l.shape)
                    else:
                        break

    x = x.sum(axis=1)
    y = y.sum(axis=1)

    return x, y


class PLADPInspectionModel:

    def solve(self, R: np.ndarray, h_plus: np.ndarray, u: np.ndarray, u0):
        """
        Returns the Linear Approximated Dynamic Programming solution by inspection.
        :param R     : Return at t
        :param h_plus: h^+_(t-1), state variable
        :param u     : u_it, old value function slopes
        :rtype       : LADPInspectionModel
        """
        e = identity(N+1)
        V_tilde_plus = zeros(N+1)
        self.k = zeros(N+1)
        for i in range(N+1):
            x, y = PLADP(R, h_plus + e[i], u, u0)
            V_tilde_plus[i], tmp = apply(u, u0, ft(h_plus + e[i], R, x, y))

        self.x, self.y = PLADP(R, h_plus, u, u0)
        self.h_plus = ft(h_plus, R, self.x, self.y)
        V_tidle, self.k = apply(u, u0, self.h_plus)
        self.delta_V_tilde = V_tilde_plus - V_tidle
        return self
