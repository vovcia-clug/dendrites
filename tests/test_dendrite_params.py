import torch
import itertools
from pprint import pprint

from config import MainConfig, create_tensorboard_writer
from dendrites.segment import DendriteSegment
from dendrites.boundary.boundary_context import BoundaryContext
from dendrites.forward.forward_context import ForwardContext
from dendrites.dendrite_engine import DendriteEngine

c = MainConfig()


class TestDendriteParams:
    def __init__(self):
        self.engine = None
        self.R0 = torch.arange(10.0, 20.0, 1.0)
        self.RA = torch.arange(0.0, 1 / 16, 1 / 64)
        self.K = torch.arange(0.0, 1 / 16, 1 / 64)
        self.GL = torch.arange(0.0, 1 / 16, 1 / 64)
        self.LEN = (5, 50)
        self.segment = None

    def gen_params(self):
        for r0, ra, k, gl, LEN in itertools.product(self.R0, self.RA, self.K, self.GL, self.LEN):
            yield {
                'Ra': ra,
                'LEN': LEN,
                'r0': r0,
                'k': k,
                'Cm': c.dendrites.CM,
                'gl': gl,
                'El': c.dendrites.EL,
                'dx': c.dendrites.DX,
                'dt': c.dendrites.DT,
                'name': f"Ra{ra}_R0{r0}_K{k}_GL{gl}_LEN{LEN}"
            }

    def test_dendrite_params(self, writer):
        for params in self.gen_params():
            self.engine = DendriteEngine(forward_context=ForwardContext(), boundary_context=BoundaryContext())
            self.segment = DendriteSegment(**params)
            self.engine.add_segment(self.segment)
            max_V, min_V, sum_V = self._test_signal_start()
            valid = torch.all(torch.isfinite(max_V)) and torch.all(torch.isfinite(min_V)) and torch.all(
                torch.isfinite(sum_V) and torch.all(max_V <= 1.0) and torch.all(min_V > -1.0))
            if not valid:
                pprint(params)
                pprint(self.segment.V)
                continue
            hparams = {'Ra': params['Ra'].item(), 'R0': params['r0'].item(), 'K': params['k'].item(),
                       'GL': params['gl'].item(), 'LEN': params['LEN']}
            writer.add_hparams(hparams, {'max_V': max_V.nan_to_num(-1, 2, -2).clamp(-3.0, 3.0).item(),
                                         'min_V': min_V.nan_to_num(-1, 2, -2).clamp(-3.0, 3.0).item(),
                                         'sum_V': sum_V.div(params['LEN']).nan_to_num(-1, 2, -2).clamp(-3.0, params[
                                             'LEN'] * 2.0).item(),
                                         'valid': valid},
                               run_name=params['name'])

    def _test_signal_start(self):
        """Test the forward propagation through segments with a signal at the start."""
        N = 100
        max_V = torch.tensor(0.0)
        min_V = torch.tensor(0.0)
        for i in range(N):
            if (i % 25) == 0:
                self.segment.V[1] = 1.0
            self.engine.forward()
            max_V = torch.max(max_V, torch.max(self.segment.V))
            min_V = torch.min(min_V, torch.min(self.segment.V))
        sum_V = torch.sum(self.segment.V)
        return max_V, min_V, sum_V
        # assert self.segment.V[-1] > 0.0
        # assert self.segment.V[-1] < 1.0


if __name__ == '__main__':
    test = TestDendriteParams()
    writer = create_tensorboard_writer()
    test.test_dendrite_params(writer)
