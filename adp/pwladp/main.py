from matplotlib import pyplot as plt

from adp.generator import GaussianGenerator
from adp.plot.slopes import FirstSlopeAx
from adp.plot.test import GrossTestPlotter
from adp.plot.wealth import FinalReturnPlotter
from adp.pwladp.trainer import ADPStrategyTrainer
from adp.strategy import ADPStrategy
from adp.value_function import PWLDynamicFunction
from parameters import S, repeat

strategy = ADPStrategy(value_function_class=PWLDynamicFunction)  # Value Function
generator = GaussianGenerator(r=0.001)                           # Scenarios Generator
trainer = ADPStrategyTrainer(gamma=0.2, generator=generator)     # Process trainer

# Plotters
plt.ion()
plotters = [
    # PWLValueFunctionPlotter(i, strategy),
    FinalReturnPlotter(trainer, lengths=(10, 50)),
    GrossTestPlotter(trainer, strategy)
]

firstSlopeAx = FirstSlopeAx(strategy)

for s in range(S):
    print(s)
    trainer.train(strategy)
    if s % repeat == 1:
        for plotter in plotters:
            plotter.draw()
        plt.pause(1)
        firstSlopeAx.plot()

plt.ioff()
