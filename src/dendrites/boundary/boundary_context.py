from dendrites.boundary.boundary_strategy_abc import BoundaryStrategyABC
from dendrites.boundary.boundary_strategy_default import BoundaryStrategyDefault


class BoundaryContext:
    def __init__(self,
                 boundary_strategy: BoundaryStrategyABC = None):
        if boundary_strategy is None:
            boundary_strategy = BoundaryStrategyDefault()
        self.boundary_strategy = boundary_strategy

    def set_boundary_strategy(self, boundary_strategy: BoundaryStrategyABC):
        self.boundary_strategy = boundary_strategy
