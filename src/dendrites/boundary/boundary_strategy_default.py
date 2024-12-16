import torch
from typing import List, Dict

from torch import Tensor

from dendrites.boundary.boundary_strategy_abc import BoundaryStrategyABC


@torch.jit.script
def _boundary(voltage_table: torch.Tensor):
    voltage_table[:, 0, 0] = voltage_table[:, 0, 1]
    voltage_table[:, 0, -1] = voltage_table[:, 0, -2]

@torch.jit.script
def boundary_torchscript(voltage_tables: Dict[int, torch.Tensor]):
    futures = torch.jit.annotate(List[torch.jit.Future[None]], [])
    for key in voltage_tables:
        voltage_table = voltage_tables[key]
        fut = torch.jit.fork(_boundary, voltage_table)
        futures.append(fut)
    for fut in futures:
        torch.jit.wait(fut)


class BoundaryStrategyDefault(BoundaryStrategyABC):
    @staticmethod
    @torch.jit.script
    def boundary(voltage_tables: Dict[int, torch.Tensor]):
        boundary_torchscript(voltage_tables)

    @staticmethod
    @torch.jit.script
    def boundary_branch(branchesM: List[Tensor], branchesL: List[Tensor], branchesR: List[Tensor]):
        V_stack = torch.stack(branchesM, dim=0)
        Vb1_stack = torch.stack(branchesL, dim=0)
        Vb2_stack = torch.stack(branchesR, dim=0)

        mean_value = (V_stack[:, -2] + Vb1_stack[:, 1] + Vb2_stack[:, 1]) / 3

        V_stack[:, 1].copy_(mean_value)
        Vb1_stack[:, 0].copy_(mean_value)
        Vb2_stack[:, 0].copy_(mean_value)

        I = -V_stack[:, 0] + mean_value
        Ib1 = -Vb1_stack[:, 1] + mean_value
        Ib2 = -Vb2_stack[:, 1] + mean_value
        I_mean = (I + Ib1 + Ib2) / 3

        V_stack[:, 0].add_(I - I_mean)
        Vb1_stack[:, 1].add_(Ib1 - I_mean)
        Vb2_stack[:, 1].add_(Ib2 - I_mean)

        for i in range(len(branchesM)):
            branchesM[i].copy_(V_stack[i])
            branchesL[i].copy_(Vb1_stack[i])
            branchesR[i].copy_(Vb2_stack[i])

    def __str__(self):
        return 'Default'


