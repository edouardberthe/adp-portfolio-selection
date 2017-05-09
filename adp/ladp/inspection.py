from numpy import identity, zeros

from adp.value_function import ValueFunction
from data import N
from ..parameters import *
from ..transition import ft


def LADP(obj, R: np.ndarray, h_plus: np.ndarray, u: np.array):
    """
    Returns the Linear Approximated Dynamic Programming solution by inspection.
    Special care is given when dealing with the variables here: h, h_plus, R are (N+1) x 1, and 
    x, y, k and l are N x 1. Subsequently, I and J are indexes of the second set of arrays.

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

    k =   u[1:] - (1 + theta) * u[0]  # Buying slopes
    l = - u[1:] + (1 - theta) * u[0]  # Selling slopes

    # Step 0
    j_star = np.argmax(k) if (k > 0).any() else None  # Best buy
    I = np.all((l > 0, h[1:] > 0), axis=0)  # Sell assets

    # Step 1: selling Sell assets
    if I.any():
        y[I] = h[1:][I]
        h[0] += (1 - theta) * y[I].sum()

    if j_star is not None:
        # Step 2: if there is a Best Buy asset, buy as much as possible with cash
        x[j_star] = h[0] / (1 + theta)
        h[0] = 0

        # Sell-to-Buy assets
        J = np.all((l <= 0, (1 - theta) / (1 + theta) * k[j_star] + l > 0, h[1:] > 0), axis=0)
        if J.any():
            # Step 3: if there is a Best-Buy and some Sell-to-Buy assets
            y[J] = h[1:][J]
            x[j_star] += (1 - theta) / (1 + theta) * y[J].sum()

    return x, y


class LADPInspectionModel(object):
    solver = LADP

    def __init__(self):
        self.x = None
        self.y = None
        self.h_plus = None
        self.ΔV = None

    def solve(self, R: np.ndarray, h_plus: np.ndarray, V: ValueFunction):
        """
        Returns the Linear Approximated Dynamic Programming solution by inspection.
        :param R     : Return at t
        :param h_plus: h^+_(t-1), state variable
        :param V     : V, old value function
        :rtype       : LADPInspectionModel
        """
        e = identity(N+1)
        V_plus = np.zeros(N+1)
        for i in range(N+1):
            x, y = self.solver(R, h_plus + e[i], V)
            V_plus[i] = V(ft(h_plus + e[i], R, x, y))

        self.x, self.y = self.solver(R, h_plus, V)
        self.h_plus = ft(h_plus, R, self.x, self.y)
        self.ΔV = V_plus - V(self.h_plus)
        return self
