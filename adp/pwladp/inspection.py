import numpy as np
from numpy import zeros

from adp.ladp.inspection import LADPInspectionModel
from adp.value_function import ValueFunction
from data import N
from parameters import M, theta


class PWLADPInspectionModel(LADPInspectionModel):

    def __init__(self):
        super().__init__()
        self.pi = None

    def solve(self, R: np.ndarray, h_plus: np.ndarray, V: ValueFunction):
        self.pi = np.argmin(V.idx(h_plus[1:]), axis=0) - 1
        return super().solve(R, h_plus, V)

    def step(self, h, V: ValueFunction):

        idx = V.idx(h[1:])
        pi = np.argmin(idx, axis=0) - 1

        # Computing max bound
        w_tidle = zeros((M, N))
        w_tidle[:-1] = V.a[1:] - V.a[:-1]
        w_tidle[-1] = float('inf')

        # Computing approximate pre-decision variables
        h_tidle = zeros((M, N))
        h_tidle[idx] = w_tidle[idx]
        h_tidle[pi, range(N)] = h[1:] - V.a[pi, range(N)]

        # Return values
        x = zeros((M, N))
        y = zeros((M, N))

        k = V.slopes - (1 + theta) * V.cash  # Buying slopes
        l = - V.slopes + (1 - theta) * V.cash  # Selling slopes

        # Step 1: selling Sell assets
        I = np.logical_and(l > 0, h_tidle > 0)  # Sell assets
        if I.any():
            y[I] = h_tidle[I]
            h[0] += (1 - theta) * y[I].sum()

        # Step 2
        while True:
            j_star_idx = np.logical_and(k > 0, h_tidle < w_tidle)
            if j_star_idx.any():
                j_star = np.unravel_index((k * j_star_idx).argmax(), (M, N))  # Best buy
                if w_tidle[j_star] - h_tidle[j_star] < h[0] / (1 + theta):
                    x[j_star] = w_tidle[j_star] - h_tidle[j_star]
                    h[0] -= (1 + theta) * x[j_star]
                else:
                    x[j_star] = h[0] / (1 + theta)
                    h[0] = 0
                    break
            else:
                break

        # Step 3: Perform sell-to-buy transactions if there is still a best buy
        while True:
            if j_star_idx.any():
                j_star = np.unravel_index((k * j_star_idx).argmax(), (M, N))
                i_star_idx = np.logical_and(l <= 0, h_tidle > 0, (1 - theta) / (1 + theta) * k[j_star] + l > 0)
                if i_star_idx.any():
                    i_star = np.unravel_index(np.where(i_star_idx, l, float('-inf')).argmax(), (M, N))
                    if h_tidle[i_star] < (1+theta)/(1-theta) * (w_tidle[j_star] - h_tidle[j_star]):
                        y[i_star] += h_tidle[i_star]
                        x[j_star] += (1-theta)/(1+theta) * h_tidle[i_star]
                    else:
                        y[i_star] += (1+theta)/(1-theta) * (w_tidle[j_star] - h_tidle[j_star])
                        x[j_star] += w_tidle[j_star] - h_tidle[j_star]
                    h_tidle[i_star] -= y[i_star]
                    h_tidle[j_star] += x[j_star]
                else:
                    break
            else:
                break
            j_star_idx = np.logical_and(k > 0, h_tidle < w_tidle)

        x = x.sum(axis=0)
        y = y.sum(axis=0)

        return x, y
