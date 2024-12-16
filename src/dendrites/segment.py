from __future__ import annotations

import torch

from config import *
from dendrites.exceptions import DendriteRadiusError


class DendriteSegment:
    def __init__(self,
                 *,
                 dx,
                 dt,
                 r0,
                 k,
                 gl,
                 El,
                 Cm,
                 Ra,
                 name,
                 LEN,
                 ):
        """
        Dendrite segment initialization. For details see paper:

        @param dx: Dx (length interval)
        @param dt: Dt (time interval)
        @param r0: Radius at start
        @param k: Tapering factor
        @param gl: Leakage factor
        @param El: Base voltage
        @param Cm: Capacitance factor
        @param Ra: Resistance factor
        @param name: Name
        @param LEN: Length
        """
        self.dx = dx.clone().detach() if isinstance(dx, torch.Tensor) else torch.tensor(dx, dtype=torch.float)
        self.dt = dt.clone().detach() if isinstance(dt, torch.Tensor) else torch.tensor(dt, dtype=torch.float)
        self.r0 = r0.clone().detach() if isinstance(r0, torch.Tensor) else torch.tensor(r0, dtype=torch.float)
        self.k = k.clone().detach() if isinstance(k, torch.Tensor) else torch.tensor(k, dtype=torch.float)
        self.gl = gl.clone().detach() if isinstance(gl, torch.Tensor) else torch.tensor(gl, dtype=torch.float)
        self.El = El.clone().detach() if isinstance(El, torch.Tensor) else torch.tensor(El, dtype=torch.float)
        self.Cm = Cm.clone().detach() if isinstance(Cm, torch.Tensor) else torch.tensor(Cm, dtype=torch.float)
        self.Ra = Ra.clone().detach() if isinstance(Ra, torch.Tensor) else torch.tensor(Ra, dtype=torch.float)
        self.name = name
        self.V = None
        self.length = LEN

    def set_length(self, new_length: int):
        if new_length == self.length:
            return
        if self.radius(new_length) <= 0:
            raise DendriteRadiusError(f"Radius at length {new_length} is {self.radius(new_length)}")
        self.length = new_length

    def radius(self, i: int):
        return self.r0 - self.k * i * self.dx

    def signal(self, i, dV):
        _i = int(i // self.dx)
        if _i < 0:
            _i = len(self.V) + _i
        assert self.length > _i >= 0
        self.V[_i].add_(dV[0]).clamp_(-1.0, 1.0)

    def D_batch(self, length: int):
        x = torch.arange(0, length, self.dx.item())
        return ((self.r0 - self.k * x) ** 2) / (2 * torch.pi * (self.r0 - self.k * x) * self.Ra)

    def log_to_tensorboard(self, writer: SummaryWriter, step):
        writer.add_scalars(f'{self.name}/V', {f'{i}': self.V[i] for i in range(self.length)}, step)

    @property
    def configuration(self):
        return dict(Ra=self.Ra,
                    r0=self.radius(self.length),
                    k=self.k,
                    Cm=self.Cm,
                    gl=self.gl,
                    El=self.El,
                    dx=self.dx,
                    dt=self.dt)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def dendrite_default_configuration(c):
    return {
        'Ra': c.dendrites.RA,
        'LEN': c.dendrites.LEN,
        'r0': c.dendrites.R0,
        'k': c.dendrites.K,
        'Cm': c.dendrites.CM,
        'gl': c.dendrites.GL,
        'El': c.dendrites.EL,
        'dx': c.dendrites.DX,
        'dt': c.dendrites.DT
    }
