from numpy.random import chisquare, multivariate_normal

from data import *
from report.config import *

figure(figsize=(10, 5))
tickers = ['AC.PA', 'ACA.PA']

lr = np.log(Gross.loc['2012':, tickers])

# Real Data
plot(*lr.as_matrix().T, 'o', label='Real Data', alpha=0.7, color='green')

# Gaussian
n_scenarios = multivariate_normal(np.zeros(2), lr.cov(), len(lr))
plot(*n_scenarios.T, 'o', label='Multivariate Gaussian Generation', color='blue')

# T- Student
nu = 2
gaussian = multivariate_normal(np.zeros(2), lr.cov(), len(lr))
chi2 = chisquare(nu, (len(lr), 1))
t_scenarios = gaussian / np.sqrt(nu / chi2) + lr.mean().as_matrix()
plot(*t_scenarios.T, '.', label='Multivariate Student-t Generation', color='red')
legend()
xlabel(tickers[0] + " log returns")
ylabel(tickers[1] + " log returns")
show()
