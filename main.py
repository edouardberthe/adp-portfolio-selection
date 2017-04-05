from imports import *

#### Illustrating Markowitz Portfolio via Efficient Frontier ####

plotCAC40()

#### Illustrating some Scenarios-Based Measures ####

models = [SemiMAD, GMD, Minimax, CVaR]

s, p = generateStudentTScenarios(500)
group = PortfolioGroup([Model(s, p, output=True).optimize().getPortfolio() for Model in models])
group.plot()

s, p = generateStudentTScenarios(100)
group = PortfolioGroup([Model(s, p, output=True, Wmin=0.05, Nmax=5).optimize().getPortfolio() for Model in models])
group.plot()

#### Illustrating BackTest ####
# The output here may not be relevant, because the number of scenarios generated and the rebalancing periods are not
# high enough. However, they are set this way to illustrate the algorithm to the user within minutes.

# Rebalancing monthly at the end of the month, covariance rolling window of 1 year, T distribution generator,
# 100 scenarios
pool = BackTestParamPool(freq='M', window=365, generator=generateStudentTScenarios, N=100)
b = BackTest(SemiMAD, pool)

# Launches the animation
animateBackTest(b)


#### Illustrating BackTestGroup ####

# Rebalancing fornightly on friday, covariance rolling window of 1 year, T distribution generator, 1000 scenarios
pool = BackTestParamPool(freq='2W-FRI', window=365, generator=generateStudentTScenarios, N=1000)
g = BackTestGroup([SemiMAD, Minimax, CVaR, EWPortfolioModel], pool)

# Launches the animation
animateBackTestGroup(g)
