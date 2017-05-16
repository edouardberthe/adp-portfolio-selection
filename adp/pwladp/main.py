from matplotlib import pyplot as plt
from numpy import random as rd

from adp.animation import PWLValueFunctionAnimation
from adp.parameters import T
from adp.pwladp.model import PWLADPModel
from adp.updater import PWLValueFunctionUpdater
from adp.value_function import PWLinearValueFunction

rd.seed(4)
V = [PWLinearValueFunction() for i in range(T)]
m = PWLADPModel()
updater = PWLValueFunctionUpdater(V, m)

if __name__ == '__main__':
    animation = PWLValueFunctionAnimation(updater, 1)
    # for s in range(S):
    #     next(updater)
    plt.show()
