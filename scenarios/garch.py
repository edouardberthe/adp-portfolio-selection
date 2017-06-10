import numpy as np
from arch import arch_model

from data import Data

D = Data.iloc[:, 1]

q = 1
ps = range(1, 10)
fits = {}
for mean in ('Constant',):
    for vol in ('GARCH', 'EGARCH', 'ARCH'):
        for dist in ('Normal', 'StudentsT'):
            for p in range(1, 20, 3):
                for q in range(1, 20, 3):
                    fits[mean, vol, dist, p, q] = arch_model(D, p=p, q=q, mean=mean, vol=vol, dist=dist).fit()

s = sorted([(v.aic, i) for (i, v) in fits.items()], reverse=True)

for x in s:
    print(x)
