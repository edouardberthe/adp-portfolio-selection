from matplotlib import pyplot as plt
from numpy import random as rd

from adp.parameters import S, T
from adp.pwladp.inspection import PWLADPInspectionModel
from adp.updater import PWLValueFunctionUpdater
from adp.value_function import PWLinearValueFunction

rd.seed(4)
V = [PWLinearValueFunction() for t in range(T)]
m = PWLADPInspectionModel()
updater = PWLValueFunctionUpdater(V, m)

if __name__ == '__main__':
    # animation = ValueFunctionAnimation(updater)
    for s in range(S):
        next(updater)
    plt.show()
