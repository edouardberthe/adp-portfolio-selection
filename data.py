import pandas as pd
import numpy as np

CAC40Tickers = ['{:s}.PA'.format(t) for t in ['AC','ACA','AI','AIR','BN','BNP','CA','CAP','CS','DG','EI','EN','ENGI','FP','FR','GLE','KER','LHN','LI','LR','MC','ML','MT','NOKIA','OR','ORA','PUB','RI','RNO','SAF','SAN','SGO','SOLB','SU','SW','TEC','UG','UL','VIE','VIV']]
# ASX100Tickers = ['{:s}.AX'.format(t) for t in ['ABC','AGL','ALQ','AWC','AMC','AMP','ANN','APA','ALL','ASX','AZJ','AST','ANZ','BOQ','BEN','BHP','BSL','BLD','BXB','CTX','CAR','CGF','CIM','CCL','COH','CBA','CPU','CWN','CSL','CSR','CYB','DXS','DMP','DOW','DUE','DLX','EVN','FXJ','FLT','FMG','GMG','GPT','GNC','HVN','HSO','HGG','ILU','IPL','IAG','IOF','IFL','JHX','JBH','LLC','LNK','MQA','MQG','MFG','MPL','MGR','NAB','NVT','NCM','NST','OSH','ORI','ORG','ORA','PPT','PRY','QAN','QBE','QUB','RHC','REA','RMD','RIO','STO','SCG','SEK','SHL','S32','SKI','SGP','SUN','SYD','TAH','TTS','TLS','SGR','TPM','TCL','TWE','VCX','VOC','WES','WFD','WBC','WPL','WOW']]
filename = "pickle/data.pkl"    # For CAC 40


def ReLoadYahooData():
    from entities.security import Security
    print("Loading data")
    Securities = [Security(t) for t in CAC40Tickers]
    Data = pd.DataFrame({s.ticker: s.data for s in Securities})
    print("Removing missing values")
    Data = Data.loc[:, Data.isnull().sum() < 0.05 * len(Data)]
    Data.dropna(inplace=True)
    print("Saving data")
    Data.to_pickle(filename)
    return Data


def FilterData(Data):
    print("Removing Outliers")
    Data.drop(Data.index[((Data - Data.mean()).abs() > 3 * Data.std()).any(axis=1)], inplace=True)
    Data.to_pickle(filename)


def LoadData():
    try:
        return pd.read_pickle(filename)
    except FileNotFoundError as e:
        print('Error:', e)
        print('Reloading data')
        return ReLoadYahooData()

Data = LoadData()[:]

# The Returns are (P[t] - P[t-1]) / P[t-1] = P[t] / P[t-1] - 1 = GrossReturns - 1
Returns = Data.pct_change()[1:]
""":type Returns: pandas.DataFrame"""

# The Gross returns are computed as Price[t] / Price[t-1]. We remove the first element because it is NaN.
Gross = Returns + 1
""":type Gross: pandas.DataFrame"""


N = len(Data.columns)
A = range(N)

# For the report
# figsize = (10, 6.18)

figsize = (20, 10)
