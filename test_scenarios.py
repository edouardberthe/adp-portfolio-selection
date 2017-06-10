import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import random as rd

from data import Data

D = Data['2009':].iloc[:, :10]            # Data
LR = np.log(D.pct_change().dropna() + 1)  # Log returns

# Simulate new returns
N = 100
index = pd.date_range(start=LR.index[-1] + pd.Timedelta('1D'), periods=N)
SimuLR = pd.DataFrame(data=rd.multivariate_normal(LR.mean(), LR.cov(), N), index=index, columns=D.columns)
SimuGross = np.exp(SimuLR)
SimuD = D.iloc[-1] * SimuGross.cumprod()

plt.ion()
pd.concat([D, SimuD])['2016':].plot(ax=plt.gca())
