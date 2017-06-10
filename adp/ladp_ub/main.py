from adp.updater import LValueFunctionUpdater
from matplotlib import pyplot as plt
from numpy import random as rd

from adp.ladp_ub import LADPUBInspectionModel
from adp.plot.animation import ValueFunctionAnimation
from adp.value_function import LinearValueFunction

rd.seed(4)
V = LinearValueFunction()
m = LADPUBInspectionModel()
updater = LValueFunctionUpdater(V, m)

if __name__ == '__main__':
    animation = ValueFunctionAnimation(updater)
    #for s in range(S):
    #    next(updater)
    plt.show()
