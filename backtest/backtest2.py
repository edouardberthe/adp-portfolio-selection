from data import Data


class Strategy:

    def __init__(self, model):
        pass


class BackTest:

    def __init__(self, strategy, freq, train_period, test_period):
        data = Data.asfreq(freq, method='pad').pct_change()[1:]
        train_data = data[train_period]
        test_data = data[test_period]
        strategy.train(train_data)
