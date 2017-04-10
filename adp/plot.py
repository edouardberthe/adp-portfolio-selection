from matplotlib import animation, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data import Data, N
from .parameters import T


class ValueFunctionAnimation(animation.FuncAnimation):

    def __init__(self, VHat):
        self.VHat = VHat
