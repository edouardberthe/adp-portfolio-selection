from gurobipy import GurobiError

import pandas as pd
from numpy import ones, zeros, array

from adp.pwladp.model import gurobiModel
from adp.value_function import SeparableValueFunction
from data import N
from parameters import T, init


class ADPStrategy:

    def __init__(self, value_function_class):
        self.value_functions = [SeparableValueFunction(value_function_class=value_function_class) for _ in range(T)]

    def __getitem__(self, item):
        return self.value_functions[item]

    def __iter__(self):
        return iter(self.value_functions)

    def score(self, gross: pd.DataFrame) -> pd.DataFrame:
        h = pd.DataFrame(index=gross.index, columns=gross.columns)
        h.iloc[0] = 0
        h.iloc[0, 0] = init

        h.iloc[1], _ = gurobiModel(ones(N+1), h.iloc[0].as_matrix(), self[0])
        for t in range(1, T):
            try:
                h.iloc[t+1], _ = gurobiModel(gross.iloc[t-1], h.iloc[t], self[t])
            except GurobiError as e:
                print('Error during testing:', e)
                h.iloc[t+1] = h.iloc[t]
        return h
