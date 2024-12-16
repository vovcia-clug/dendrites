from config import MainConfig, create_tensorboard_writer
from dendrites.dendrite_engine import DendriteEngine
from dendrites.segment import dendrite_default_configuration


class DendriteApp:
    def __init__(self, c: MainConfig):
        self.c = c
        self.dendrite_engine = DendriteEngine(self.c)
        self.dendrite_configuration = dendrite_default_configuration(self.c)
        self.dendrite_main = self.dendrite_engine.create_segment(**self.dendrite_configuration, name="D_main")
        self.dendrite_branch_L = self.dendrite_engine.create_segment(**self.dendrite_configuration, name="D_branch_L")
        self.dendrite_branch_R = self.dendrite_engine.create_segment(**self.dendrite_configuration, name="D_branch_R")
        self.dendrite_engine.add_branch(self.dendrite_main, self.dendrite_branch_L, self.dendrite_branch_R)

    def forward(self):
        self.dendrite_engine.forward()

    def log_to_tensorboard(self, writer, step):
        if writer is not None and step is not None:
            self.dendrite_engine.log_to_tensorboard(writer, step)


if __name__ == '__main__':
    c = MainConfig()
    app = DendriteApp(c)
    writer = create_tensorboard_writer()
    N = 1000
    for step in range(N):
        app.forward()
        app.log_to_tensorboard(writer, step)
        if (step % 100) == 0:
            print(f"Step: {step}, signaling..")
            app.dendrite_main.signal(1, [1/8])
    print(f"Done, check tensorboard run {writer.run} (tensorboard --logdir=runs)")

