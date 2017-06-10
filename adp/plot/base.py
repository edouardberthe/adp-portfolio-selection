from matplotlib import pyplot as plt

from parameters import figsize


class PlotterProcess:

    colors = 'bgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykw'

    def __init__(self):
        super().__init__()
        self.fig = plt.figure(figsize=figsize)
