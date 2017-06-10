from adp.pwladp.trainer import ADPStrategyTrainer
from adp.strategy import ADPStrategy
from adp.value_function import PWLDynamicFunction
from data import Data, Returns
from markowitz.model import Markowitz
from parameters import S

# Train
Returns_train = Returns['2015']

# Test
Returns_test = Data.asfreq('W-FRI', method='pad').pct_change()[1:]['2016']


strategy = ADPStrategy(value_function_class=PWLDynamicFunction)

# Process trainer
trainer = ADPStrategyTrainer(strategy)
test_perf = []

for s in range(S):
    print(s)
    next(trainer)

adp_return = strategy.score(Returns_test+1)

markowitz = Markowitz(mean=Returns_train.mean(), cov=Returns_train.cov())
port = markowitz.optimize().getPortfolio()

marko_return = ((Returns_test + 1).prod() * port).sum() - 1
print('Markowitz return: {:.1%}'.format(marko_return))
