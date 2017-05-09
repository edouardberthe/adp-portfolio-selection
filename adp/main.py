from numpy import random as rd
from matplotlib import pyplot as plt

from adp.animation import ValueFunctionAnimation
from adp.generator import GaussianGenerator
from adp.ladp.inspection import LADPInspectionModel
from adp.ladp_ub.inspection import LADPUBInspectionModel
from adp.parameters import S
from adp.updater import ValueFunctionUpdater
from adp.value_function import LinearValueFunction

rd.seed(4)
V = LinearValueFunction()
m = LADPInspectionModel()
gen = GaussianGenerator()
updater = ValueFunctionUpdater(V, m, gen)

if __name__ == '__main__':
    animation = ValueFunctionAnimation(updater)
    # for s in range(S):
        # next(updater)
    plt.show()
