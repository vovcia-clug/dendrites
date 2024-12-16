from dendrites.forward.forward_strategy_abc import ForwardStrategyABC
from dendrites.forward.forward_strategy_default import ForwardStrategyDefault


class ForwardContext:
    def __init__(self, c, strategy: ForwardStrategyABC = None, *args, **kwargs):
        self.c = c
        if strategy is None:
            strategy = ForwardStrategyDefault(c, *args, **kwargs)
        self.strategy = strategy

    def set_strategy(self, strategy: ForwardStrategyABC):
        self.strategy = strategy

    def forward(self, *args, **kwargs):
        self.strategy.forward(*args, **kwargs)
