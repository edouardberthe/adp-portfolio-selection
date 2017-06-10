from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import pacf, acf

from data import *

d = Data.iloc[:, 0]

plt.ion()
d.plot(fig=plt.figure())

acf(d)
pacf(d)

model = ARIMA(d, order=(3, 1, 1))
model_fit = model.fit()

model_fit.resid.plot()
plt.show()
