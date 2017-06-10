from adp.updater import LValueFunctionUpdater
from matplotlib import pyplot as plt
from numpy import random as rd

from adp.ladp import LADPModel
from adp.plot.animation import ValueFunctionAnimation
from adp.value_function import LinearValueFunction

rd.seed(4)
# V = [PWLinearValueFunction() for t in range(T)]
V = LinearValueFunction()
m = LADPModel()
updater = LValueFunctionUpdater(V, m)

if __name__ == '__main__':
    animation = ValueFunctionAnimation(updater)
    #for s in range(S):
    #    next(updater)
    plt.show()
