from abc import ABC, abstractmethod
from config import *


class ForwardStrategyABC(ABC):
    @staticmethod
    @abstractmethod
    def forward(*args, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def cache_clear(self):
        pass
