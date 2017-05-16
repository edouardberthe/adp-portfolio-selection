from matplotlib import pyplot as plt
from numpy import random as rd

from adp.animation import ValueFunctionAnimation
from adp.ladp_ub import LADPUBInspectionModel
from adp.updater import LValueFunctionUpdater
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
