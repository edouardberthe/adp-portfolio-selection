import pandas as pd

from parameters import periods
from report.config import *

bond = pd.read_pickle('pickle/bond.pkl')

bond['2005':].plot(figsize=figsize)
for (name, start, middle, end, height) in periods:
    annual_yield = bond[start:end].mean()
    weekly_yield = (1 + annual_yield)**(1/52) - 1
    print('{:s}: {:.3f}%, {:.3f}%'.format(name, annual_yield, weekly_yield))

xlabel('Time')
ylabel('Bond Yield (\%)')
show()
