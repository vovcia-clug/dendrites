from abc import ABC, abstractmethod
from typing import List, Dict

from torch import Tensor

from dendrites.voltage_cache_table import VoltageCacheTable


class BoundaryStrategyABC(ABC):
    @abstractmethod
    def boundary(self, voltages: Dict[int, VoltageCacheTable]):
        pass

    @abstractmethod
    def boundary_branch(self, branchesM: List[Tensor], branchesL: List[Tensor], branchesR: List[Tensor]):
        pass
