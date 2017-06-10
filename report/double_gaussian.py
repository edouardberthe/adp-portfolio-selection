from numpy.random import chisquare, multivariate_normal

from data import *
from report.config import *

figure(figsize=(10, 6))
tickers = ['AC.PA', 'ACA.PA']

lr = np.log(Gross.loc['2012':, tickers])

scatter(*lr.as_matrix().T, label='Log returns')
nu = 3
gaussian = multivariate_normal(np.zeros(2), lr.cov(), len(lr))
chi2 = chisquare(nu, (len(lr), 1))
scenarios = gaussian / np.sqrt(nu / chi2) + lr.mean().as_matrix()
scatter(*scenarios.T, label='Multivariate Student-t Generation', color='red')
legend()
xlabel(tickers[0] + " log returns")
ylabel(tickers[1] + " log returns")
show()
