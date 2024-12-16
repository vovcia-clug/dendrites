from config import *
from dendrites.boundary.boundary_context import BoundaryContext
from dendrites.dendrite_engine import DendriteEngine
from dendrites.forward.forward_context import ForwardContext
from dendrites.log_matshow import log_matshow
from dendrites.segment import DendriteSegment, dendrite_default_configuration


SIGNAL1 = 50.0
SIGNAL2 = 30.0
LEN = 5


class TestDendriteBranching:
    def __init__(self):
        c = MainConfig()
        self.engine = DendriteEngine(c, forward_context=ForwardContext(c), boundary_context=BoundaryContext())
        configuration = dendrite_default_configuration(c)
        configuration["LEN"] = LEN
        self.segment = DendriteSegment(name="D0", **configuration)
        self.engine.add_segment(self.segment)
        self.branch_L = DendriteSegment(**self.segment.configuration, name="L", LEN=LEN)
        self.branch_R = DendriteSegment(**self.segment.configuration, name="R", LEN=LEN)
        self.engine.add_segment(self.branch_L)
        self.engine.add_segment(self.branch_R)
        self.engine.add_branch(self.segment, self.branch_L, self.branch_R)
        self.step = 0

    def forward(self, writer, potentials):
        # self.branch_L.V[LEN // 2] = torch.sin(torch.tensor(self.step / SIGNAL1)).clamp_max(0.0) * (self.step < 1000)
        self.branch_R.V[2] = torch.cos(torch.tensor(self.step / SIGNAL2)).clamp(0.0, 1.0) * (self.step < 1500)
        self.engine.forward()
        self.engine.log_to_tensorboard(writer, self.step)
        potentials[0, self.step, :] = self.segment.V
        potentials[1, self.step, :] = self.branch_L.V
        potentials[2, self.step, :] = self.branch_R.V
        self.step += 1


if __name__ == "__main__":
    writer = create_tensorboard_writer()
    test = TestDendriteBranching()
    N = 2500
    potentials = torch.zeros((3, N, LEN))
    for i in range(N):
        test.forward(writer, potentials)
    log_matshow(potentials, writer, 0)
