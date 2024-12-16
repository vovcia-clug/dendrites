import functools
from typing import List

import torch

from config import *
from dendrites.forward.forward_strategy_abc import ForwardStrategyABC


class ForwardStrategyDefault(ForwardStrategyABC):
    __slots__ = ('dx', 'dt', 'gl', 'El', 'Cm')

    def __init__(self, c, *, dx=None, dt=None, gl=None, El=None, Cm=None, **kwargs):
        super().__init__()
        if dx is None:
            dx = c.dendrites.DX
        if dt is None:
            dt = c.dendrites.DT
        if gl is None:
            gl = c.dendrites.GL
        if El is None:
            El = c.dendrites.EL
        if Cm is None:
            Cm = c.dendrites.CM
        self.dx = dx if isinstance(dx, torch.Tensor) else torch.tensor(dx, dtype=torch.float)
        self.dt = dt if isinstance(dt, torch.Tensor) else torch.tensor(dt, dtype=torch.float)
        self.gl = gl if isinstance(gl, torch.Tensor) else torch.tensor(gl, dtype=torch.float)
        self.El = El if isinstance(El, torch.Tensor) else torch.tensor(El, dtype=torch.float)
        self.Cm = Cm if isinstance(Cm, torch.Tensor) else torch.tensor(Cm, dtype=torch.float)

    def cache_clear(self):
        pass

    def forward(self, voltage_tables):
        for length, table in voltage_tables.items():
            self.update_voltages(table.data, self.dt, self.Cm, self.dx, self.gl, self.El)

    @staticmethod
    @torch.jit.script
    def update_voltages(data: torch.Tensor,
                        dt: torch.Tensor,
                        Cm: torch.Tensor,
                        dx: torch.Tensor,
                        gl: torch.Tensor,
                        El: torch.Tensor):
        delta_v = (dt / Cm) * (
            data[:, 1, 1:-1] / dx ** 2 * (data[:, 0, 0:-2] - 2 * data[:, 0, 1:-1] + data[:, 0, 2:]) - gl * (
                data[:, 0, 1:-1] - El)
        )
        data[:, 0, 1:-1] += delta_v

    def __str__(self):
        return "Default"
