import pandas as pd

from parameters import periods
from report.config import *

cac = pd.read_pickle('pickle/cac.pkl')

cac.plot(figsize=figsize)

for (name, start, _, end, height) in periods:
    annotate('', xy=(start, height), xytext=(end, height), arrowprops=dict(arrowstyle='<->'))
    plot([start, start], [height-300, height+300], 'black')
    plot([end, end], [height-300, height+300], 'black')
    middle = pd.DatetimeIndex(start=start, end=end, freq='D')
    middle = middle[len(middle)//4]
    text(middle, height+100, name)

xlabel('Time')
ylabel('CAC 40 Index')
show()

