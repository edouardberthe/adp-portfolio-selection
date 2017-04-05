import pandas as pd
import numpy as np

CAC40Tickers = ['{:s}.PA'.format(t) for t in ['AC','ACA','AI','AIR','BN','BNP','CA','CAP','CS','DG','EI','EN','ENGI','FP','FR','GLE','KER','LHN','LI','LR','MC','ML','MT','NOKIA','OR','ORA','PUB','RI','RNO','SAF','SAN','SGO','SOLB','SU','SW','TEC','UG','UL','VIE','VIV']]


def ReLoadYahooData():
    """"""
    from entities import Security
    print("Loading data")
    Securities = [Security(t) for t in CAC40Tickers]
    Data = pd.DataFrame({s.ticker: s.data for s in Securities})
    print("Removing missing values")
    Data = Data.loc[:, Data.isnull().sum() < 0.05 * len(Data)]
    Data.dropna(inplace=True)
    print("Saving data")
    Data.to_pickle("pickle/data.pkl")
    return Data


def FilterData(Data):
    print("Removing Outliers")
    Data.drop(Data.index[((Data - Data.mean()).abs() > 3 * Data.std()).any(axis=1)], inplace=True)
    Data.to_pickle("pickle/data.pkl")
    return Data


def LoadData():
    return pd.read_pickle("pickle/data.pkl")


Data = LoadData().iloc[:, :5]

# The Gross returns are computed as Price[t] / Price[t-1]. We remove the first element because it is NaN.
GrossReturns = (Data / Data.shift())[1:]
""":type GrossReturns: pandas.DataFrame"""
# We remove the values too extreme to be simply outliers (a daily return of more than 50% is VERY unlikely..).
GrossReturns.drop(GrossReturns.index[(GrossReturns > 1.5).any(axis=1)], inplace=True)

# The Returns are (P[t] - P[t-1]) / P[t-1] = P[t] / P[t-1] - 1 = GrossReturns - 1
Returns = GrossReturns - 1
""":type Returns: pandas.DataFrame"""

DailyMeanReturns = Returns.mean()
MeanReturns = DailyMeanReturns * 252

DailyVarReturns = Returns.var()
VarReturns = DailyVarReturns * 252

DailyVolReturns = Returns.std()
VolReturns = DailyVolReturns * np.sqrt(252)

CovReturns = Returns.cov() * 252

A = range(len(Data.columns))

# For the report
# figsize = (10, 6.18)

figsize = (20, 10)