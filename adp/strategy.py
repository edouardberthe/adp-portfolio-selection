from gurobipy import GurobiError

import pandas as pd
from numpy import ones, zeros

from adp.pwladp.model import gurobiModel
from adp.value_function import PWLValueFunction
from data import N
from parameters import T, init


class ADPStrategy(list):

    def __init__(self, value_function_class):
        super().__init__(PWLValueFunction(value_function_class=value_function_class) for t in range(T))

    def score(self, gross: pd.DataFrame):
        hp = zeros(N + 1)
        hp[0] = init
        hp, _ = gurobiModel(ones(N+1), hp, self[0])
        for t in range(1, T):
            try:
                hp, _ = gurobiModel(gross.iloc[t-1], hp, self[t])
            except GurobiError as e:
                print('Error during testing:', e)
        return (hp * gross.iloc[T-1]).sum()
