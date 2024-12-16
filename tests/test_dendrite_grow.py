import unittest

from config import MainConfig
from dendrites.boundary.boundary_context import BoundaryContext
from dendrites.dendrite_engine import DendriteEngine
from dendrites.forward.forward_context import ForwardContext
from dendrites.segment import dendrite_default_configuration, DendriteSegment


LEN = 5


class TestDendriteGrow(unittest.TestCase):
    def setUp(self):
        self.c = MainConfig()
        self.engine = DendriteEngine(c=self.c, forward_context=ForwardContext(c=self.c), boundary_context=BoundaryContext())
        configuration = dendrite_default_configuration(c=self.c)
        configuration["LEN"] = LEN
        self.segment = DendriteSegment(name="D0", **configuration)
        self.engine.add_segment(self.segment)

    def test_grow(self):
        orig_len = self.segment.length
        new_len = self.engine.grow(self.segment)
        self.assertEqual(new_len, orig_len + 1)

    def test_grow_voltage_table(self):
        self.segment.V.fill_(1.0)
        self.assertEqual(self.segment.V[0], 1.0)
        self.assertEqual(self.segment.V[1], 1.0)
        self.assertEqual(self.segment.V[-1], 1.0)
        self.engine.grow(self.segment)
        self.assertEqual(self.segment.V[0], 1.0)
        self.assertEqual(self.segment.V[1], 1.0)
        self.assertEqual(self.segment.V[-1], 1.0)

    def test_grow_forward(self):
        self.segment.V.fill_(1.0)
        self.engine.grow(self.segment)
        self.engine.forward()
        self.assertTrue((self.segment.V > 0.5).all())


if __name__ == '__main__':
    unittest.main()
