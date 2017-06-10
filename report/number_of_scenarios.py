from matplotlib import pyplot as plt

from adp.generator import GaussianGenerator
from adp.plot.slopes import FirstSlopeAx
from adp.plot.test import GrossTestPlotter
from adp.plot.value_function import PWLValueFunctionPlotter
from adp.plot.wealth import FinalReturnPlotter
from adp.pwladp.trainer import ADPStrategyTrainer
from adp.strategy import ADPStrategy
from adp.value_function import PWLDynamicFunction
from data import Data
from parameters import S, periods, repeat, freq

(period, start, middle, end, _) = periods[3]

Gross_test = (Data.asfreq(freq, method='pad').pct_change() + 1)[1:][middle:end]
generator = GaussianGenerator(start=start, end=middle)
strategy = ADPStrategy(value_function_class=PWLDynamicFunction)
trainer = ADPStrategyTrainer(gamma=0.5, generator=generator)

# Plot
plt.ion()
plotters = [PWLValueFunctionPlotter(i, strategy) for i in []] \
           + [FinalReturnPlotter(trainer, lengths=(10, 50)),
              # FinalPositionsPlotter(trainer),
              # SlopesNumberPlotter(V),
              # MeanSlopesPlotter(V),
              # MeanBreaksPlotter(V)
              GrossTestPlotter(trainer)
              ]

firstSlopeAx = FirstSlopeAx(strategy)

for s in range(S):
    if s % 10 == 0:
        print(s)
    trainer.train(strategy)

print(strategy.score(Gross_test))


# Plotters
while trainer.counter < S:
    print(trainer.counter)
    trainer.train(strategy)
    if trainer.counter % repeat == 1:
        for plotter in plotters:
            plotter.draw()
        plt.pause(0.001)
        firstSlopeAx.plot()

plt.ioff()
