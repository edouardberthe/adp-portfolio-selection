import pandas as pd
from pandas_datareader import data as web
from pandas_datareader._utils import RemoteDataError


class Security(object):

    def __init__(self, ticker, name=None, field='Adj Close'):
        self.ticker = ticker
        self.name = name
        self.field = field
        self._data = None

    def __str__(self):
        return self.name if self.name is not None else self.ticker

    def __repr__(self):
        return self.ticker

    def load(self, start='2005-01-01', end=None, data_provider='yahoo'):
        if self._data is None:
            print('Loading data for', self)
            try:
                self._data = web.DataReader(self.ticker, data_provider, start, end)[self.field]
            except RemoteDataError as e:
                print('Data not loaded:', e)

    @property
    def data(self):
        if self._data is None:
            self.load()
        return self._data

    def returns(self):
        return self.data / self.data.shift() - 1

    def mean_return(self):
        return self.returns.mean()


class Group(list):

    def __init__(self, name, securities):
        super().__init__(securities)
        self.name = name
        self._data = None

    @property
    def data(self):
        if self._data is None:
            try:
                self._data = pd.read_pickle(self.filename)
            except IOError:
                self.load()
        return self._data

    def load(self):
        df = pd.DataFrame({s.ticker: s.data for s in self})
        # We remove the stocks where more that 5% data is missing
        df = df.loc[:,df.isnull().sum() < 0.05 * len(df)]
        df.dropna(inplace=True)

    def save(self):
        self.data.to_pickle(self.filename)

    @property
    def filename(self):
        return "pickle/{:s}".format(self.name)

