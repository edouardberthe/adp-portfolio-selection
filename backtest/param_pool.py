import pandas as pd

from generator import ScenarioGenerator


class BackTestParamPool(object):
    """Represents a set of parameters for a BackTest."""

    def __init__(self, freq, window, generator: ScenarioGenerator, N: int, reconfigure: bool = True):
        """
        :type freq: string | pandas.DateOffset - Frequency of rebalancing, with anchor for exact rebalancing date, ex:
                                                    - weekly: 'W-MON', 'W-TUE', etc.
                                                    - fornightly: '2W-MON', '2W-FRI', etc.
                                                    - monthly: 'M' (rebalacing last day of the month)
                                                    - annually: 'A-JAN', 'A-DEC', etc.
        :type window:      int                 - Time window (in days) to compute the rolling mean / covariance matrix
        :type generator:   function            - Scenarios generator
        :type N:           int                 - Number of scenarios to generate at each rebalancing date
        :type reconfigure: bool                - Do we have to use the reconfigure method in the Models?
        """
        self.freq = freq
        self.window = pd.Timedelta(days=window)
        self.generator = generator
        self.N = N
        self.reconfigure = reconfigure

    def __str__(self):
        return f"Params: {self.N:d} scenarios - {self.freq:s} re-balancing - Rolling window: {self.window.days:d} days - {self.generator:s}"

    def __repr__(self):
        return "<BackTest Parameters Pool {:s}".format(self)