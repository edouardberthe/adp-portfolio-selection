import pandas as pd

from entities.security import Security

CAC40Tickers = [f'{t:s}.PA' for t in [
    'AC', 'ACA', 'AI', 'AIR', 'BN', 'BNP', 'CA', 'CAP', 'CS', 'DG', 'EN', 'ENGI', 'FP', 'FR', 'GLE', 'KER', 'LHN', 'LI',
    'LR', 'MC', 'ML', 'MT', 'NOKIA', 'OR', 'ORA', 'PUB', 'RI', 'RNO', 'SAF', 'SAN', 'SGO', 'SOLB', 'SU', 'SW', 'UG',
    'VIE', 'VIV']]
# ASX100Tickers = ['{:s}.AX'.format(t) for t in ['ABC','AGL','ALQ','AWC','AMC','AMP','ANN','APA','ALL','ASX','AZJ','AST','ANZ','BOQ','BEN','BHP','BSL','BLD','BXB','CTX','CAR','CGF','CIM','CCL','COH','CBA','CPU','CWN','CSL','CSR','CYB','DXS','DMP','DOW','DUE','DLX','EVN','FXJ','FLT','FMG','GMG','GPT','GNC','HVN','HSO','HGG','ILU','IPL','IAG','IOF','IFL','JHX','JBH','LLC','LNK','MQA','MQG','MFG','MPL','MGR','NAB','NVT','NCM','NST','OSH','ORI','ORG','ORA','PPT','PRY','QAN','QBE','QUB','RHC','REA','RMD','RIO','STO','SCG','SEK','SHL','S32','SKI','SGP','SUN','SYD','TAH','TTS','TLS','SGR','TPM','TCL','TWE','VCX','VOC','WES','WFD','WBC','WPL','WOW']]

FILENAME = "data/data.csv"


def reload_yahoo_data(filename: str = FILENAME):
    print("Loading data")
    securities = [Security(t) for t in CAC40Tickers]
    df: pd.DataFrame = pd.DataFrame({s.ticker: s.data for s in securities})
    print("Removing missing values")
    df = df.loc[:, df.isnull().sum() < 0.05 * len(df)]
    df.dropna(inplace=True)
    print("Saving data")
    df.to_csv(filename)
    return df


def filter_data(df: pd.DataFrame, filename: str = FILENAME):
    print("Removing Outliers")
    df.drop(df.index[((df - df.mean()).abs() > 3 * df.std()).any(axis=1)], inplace=True)
    df.to_csv(filename)


def load_data(cache: bool = True, filename: str = FILENAME):
    if cache:
        try:
            return pd.read_csv(filename)
        except FileNotFoundError as e:
            print(f'Error: {e}')
            print('Reloading data')
            return reload_yahoo_data()
    else:
        return reload_yahoo_data()


Data = load_data(cache=True)[:]

# The Returns are (P[t] - P[t-1]) / P[t-1] = P[t] / P[t-1] - 1 = GrossReturns - 1
# We remove the first element because it is NaN.
Returns = Data.pct_change()[1:]
""":type Returns: pandas.DataFrame"""

# The Gross returns are computed as Price[t] / Price[t-1]
Gross: pd.DataFrame = Returns
""":type Gross: pandas.DataFrame"""

N = len(Data.columns)
A = range(N)

# For the report
# figsize = (10, 6.18)

figsize = (20, 10)
