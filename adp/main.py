from matplotlib import pyplot as plt
from numpy import random as rd

from adp.animation import ValueFunctionAnimation
from adp.ladp import LADPInspectionModel, LADPModel
from adp.parameters import S, T
from adp.pwladp.inspection import PWLADPInspectionModel
from adp.updater import PWLValueFunctionUpdater, LValueFunctionUpdater
from adp.value_function import PWLinearValueFunction, LinearValueFunction

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
