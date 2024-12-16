from .dendrites import DendritesConfig
from .tensorboard import create_tensorboard_writer


class MainConfig:
    dendrites = DendritesConfig()

    def __setattr__(self, *args, **kwargs):
        raise AttributeError("Cannot change MainConfig attributes")


