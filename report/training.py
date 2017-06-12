import os
import pickle

from adp.generator import StudentTGenerator
from adp.pwladp.trainer import ADPStrategyTrainer
from adp.strategy import ADPStrategy
from adp.value_function import PWLDynamicFunction
from data import Data
from parameters import S, freq, gammas, k, periods, perf_dir_name

key = 'UU'
(period, start, middle, end, r) = periods[key]
print('Period', period)

# Test Data
Gross_test = (Data.asfreq(freq, method='pad').pct_change() + 1)[1:][middle:end]
Gross_test.insert(0, 'r', 1 + r)
# Scenarios Generator
type = 'student'
generator = StudentTGenerator(r=r, nu=2, start=start, end=middle)

for m in (5,):
    print('m =', m)
    for gamma in gammas:
        print('\tGamma', gamma)
        results = {}
        strategy = ADPStrategy(value_function_class=PWLDynamicFunction)
        trainer = ADPStrategyTrainer(gamma=gamma, generator=generator)
        for s in range(S):
            trainer.train(strategy)
            if s % 100 == 0:
                print('\t\t', s)
            if s % 10 == 0:
                results[s] = strategy.score(Gross_test).iloc[-1].sum()
        dirname = perf_dir_name.format(type, S, k, key, m, gamma * 10)
        os.makedirs(dirname, exist_ok=True)
        pickle.dump(file=open(dirname + "strategy", "wb"), obj=strategy)
        pickle.dump(file=open(dirname + "memory",   "wb"), obj=trainer.memory)
        pickle.dump(file=open(dirname + "test",     "wb"), obj=results)
