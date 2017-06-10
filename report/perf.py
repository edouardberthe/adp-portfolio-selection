import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from adp.generator import GaussianGenerator
from adp.pwladp.trainer import ADPStrategyTrainer
from adp.strategy import ADPStrategy
from adp.value_function import PWLDynamicFunction
from data import Data
from parameters import S, periods, freq

results = {}

for (period, start, middle, end, r) in periods:
    print('Period', period)

    # Test Data
    Gross_test = (Data.asfreq(freq, method='pad').pct_change() + 1)[1:][middle:end]
    Gross_test.insert(0, 'r', 1 + r)
    # Scenarios Generator
    generator = GaussianGenerator(r=r, start=start, end=middle)

    for gamma in np.arange(0, 1.1, 0.2):
        strategy = ADPStrategy(value_function_class=PWLDynamicFunction)
        trainer = ADPStrategyTrainer(gamma=gamma, generator=generator)
        for s in range(S):
            if s % 10 == 0:
                print(s)
            trainer.train(strategy)

        dirname = "pickle/results/gamma_{:1f}_period_{}/".format(gamma, period.replace(' ', ''))
        os.makedirs(dirname, exist_ok=True)
        pickle.dump(file=open(dirname + "strategy", "wb"), obj=strategy)
        pickle.dump(file=open(dirname + "memory", "wb"),   obj=trainer.memory)
        results[period, gamma] = strategy.score(Gross_test)

plt.ioff()

