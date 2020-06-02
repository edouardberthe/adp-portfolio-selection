from adp.cvar import CVaR
from backtest.animation import animate_backtest, animate_backtest_group
from backtest.group import BackTestGroup
from backtest.param_pool import BackTestParamPool
from backtest.backtest import BackTest
from entities.model import EWPortfolioModel
from entities.portfolio import PortfolioGroup
from generator import TGenerator
from plot_test.markowitz import plotCAC40
from scenarios_based.models import SemiMAD, GMD, Minimax

plotCAC40()

# Illustrating some Scenarios-Based Measures

models = [SemiMAD, GMD, Minimax, CVaR]

generator = TGenerator(nu=3)
s, p = generator.generate(500)
group = PortfolioGroup([Model(s, p, output=True).optimize().getPortfolio() for Model in models])
group.plot()

s, p = generator.generate(100)
group = PortfolioGroup([Model(s, p, output=True, Wmin=0.05, Nmax=5).optimize().getPortfolio() for Model in models])
group.plot()

# Illustrating BackTest
# The output here may not be relevant, because the number of scenarios generated and the rebalancing periods are not
# high enough. However, they are set this way to illustrate the algorithm to the user within minutes.

# Monthly re-balancing (anchored at the end of the month), covariance rolling window of 1 year, Student-t generator
# 100 scenarios
pool = BackTestParamPool(freq='M', window=365, generator=generator, N=100)
b = BackTest(SemiMAD, pool)

# Launches the animation
animate_backtest(b)


# Illustrating BackTestGroup

# Fortnight re-balancing on friday, covariance rolling window of 1 year, T distribution generator, 1000 scenarios
pool = BackTestParamPool(freq='2W-FRI', window=365, generator=generator, N=1000)
g = BackTestGroup([SemiMAD, Minimax, CVaR, EWPortfolioModel], pool)

# Launches the animation
animate_backtest_group(g)
