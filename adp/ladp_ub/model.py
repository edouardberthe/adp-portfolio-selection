from adp.ladp.model import LADPModel
from adp.parameters import w0
from data import N


class LADPUBModel(LADPModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._UBConstrs = [self.addConstr(self._x[i] - self._y[i] <= 0) for i in range(N)]

    def set(self, R, h_plus, u):
        super().set(R, h_plus, u)
        for i in range(N):
            self._UBConstrs[i].RHS = w0 - self._h[i+1]
