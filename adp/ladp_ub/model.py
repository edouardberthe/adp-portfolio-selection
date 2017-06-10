from numpy import array

from adp.ladp.model import LADPModel
from data import N
from parameters import w0


class LADPUBModel(LADPModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._UBConstrs = [self.addConstr(self._x[i] - self._y[i] <= 0) for i in range(N)]

    def set(self, R, h_plus, u):
        super().set(R, h_plus, u)
        for i in range(N):
            self._UBConstrs[i].RHS = w0 - self._h[i+1]

    @property
    def deltaV(self):
        deltaV = super().deltaV
        deltaV[1:] -= array([cstr.Pi for cstr in self._UBConstrs]) * self._R[1:]
        return deltaV
