import unittest
from pprint import pprint

from config import MainConfig
from dendrites.segment import DendriteSegment, dendrite_default_configuration
from dendrites.voltage_cache_table import VoltageCacheTable
from dendrites.boundary.boundary_context import BoundaryContext
from dendrites.forward.forward_context import ForwardContext
from dendrites.dendrite_engine import DendriteEngine
from dendrites.exceptions import DendriteCacheFull


class TestVoltageCacheTable(unittest.TestCase):

    def setUp(self):
        """Create a VoltageCacheTable for testing"""
        self.length = 10
        self.initial_size = 5
        self.cache_table = VoltageCacheTable(self.length, self.initial_size)

    def test_initial_setup(self):
        """Test the initial setup of the VoltageCacheTable."""
        self.assertEqual(self.cache_table.total, self.initial_size)
        self.assertEqual(self.cache_table.free, self.initial_size)
        self.assertEqual(self.cache_table.data.shape, (self.initial_size, 2, self.length))

    def test_reserve_and_free_slice(self):
        """Test reserving and freeing a slice in the voltage cache."""
        segment = DendriteSegment(dx=1.0, dt=1.0, r0=5.0, k=0.01, gl=1.0, El=-65.0, Cm=1.0, Ra=100.0, name="seg1",
                                  LEN=self.length)
        slice_id = self.cache_table.reserve_slice(segment)

        self.assertEqual(self.cache_table.free, self.initial_size - 1)
        self.assertTrue(slice_id in self.cache_table.reserved)

        # Free the reserved slice
        self.cache_table.free_slice(slice_id)
        self.assertEqual(self.cache_table.free, self.initial_size)

    def test_extend_cache(self):
        """Test extending the voltage cache table."""
        self.cache_table.extend(5)
        self.assertEqual(self.cache_table.total, self.initial_size + 5)
        self.assertEqual(self.cache_table.free, self.initial_size + 5)
        self.assertEqual(self.cache_table.data.shape, (self.initial_size + 5, 2, self.length))

    def test_cache_full_exception(self):
        """Test that a DendriteCacheFull exception is raised when cache is full."""
        for _ in range(self.initial_size):
            segment = DendriteSegment(dx=1.0, dt=1.0, r0=5.0, k=0.01, gl=1.0, El=-65.0, Cm=1.0, Ra=100.0, name="seg",
                                      LEN=self.length)
            self.cache_table.reserve_slice(segment)

        with self.assertRaises(DendriteCacheFull):
            segment = DendriteSegment(dx=1.0, dt=1.0, r0=5.0, k=0.01, gl=1.0, El=-65.0, Cm=1.0, Ra=100.0,
                                      name="seg_overflow", LEN=self.length)
            self.cache_table.reserve_slice(segment)


class TestDendriteEngine(unittest.TestCase):

    def setUp(self):
        """Create a DendriteEngine and default context for testing."""
        self.c = MainConfig()
        self.engine = DendriteEngine(c=self.c, forward_context=ForwardContext(c=self.c), boundary_context=BoundaryContext())
        self.segment_length = dendrite_default_configuration(self.c)["LEN"]
        self.segment = DendriteSegment(**dendrite_default_configuration(self.c), name="D0")
        self.engine.add_segment(self.segment)

    def test_add_segment(self):
        """Test adding a dendrite segment to the engine."""
        self.assertIn(self.segment, self.engine.segments)
        self.assertEqual(self.engine.count, 1)

    def test_grow_segment(self):
        """Test growing a segment in the engine."""
        original_length = self.segment.length

        # Grow the segment by 1 unit
        new_length = self.engine.grow(self.segment)
        self.assertEqual(new_length, original_length + 1)
        self.assertEqual(self.segment.length, new_length)
        self.assertEqual(len(self.segment.V), new_length)

    def test_segment_voltage_cache(self):
        self.segment.V.fill_(1.0)
        self.assertEqual(self.segment.V[0], 1.0)
        self.assertEqual(self.segment.V[1], 1.0)
        self.assertEqual(self.segment.V[-1], 1.0)
        new_length = self.engine.grow(self.segment)
        self.assertEqual(self.segment.V[0], 1.0)
        self.assertEqual(self.segment.V[1], 1.0)
        self.assertEqual(self.segment.V[-1], 1.0)

    def test_forward(self):
        """Test the forward propagation through segments."""
        self.segment.signal(2, [1.0])

        # Run forward propagation
        self.engine.forward()

        # Check that voltage has been updated
        self.assertNotEqual(self.segment.V[2], 0.0)

    def test_signal_start(self):
        """Test the forward propagation through segments with a signal at the start."""

        N = 100
        for i in range(N):
            self.segment.V[1] = 1.0
            self.engine.forward()

        self.assertGreater(self.segment.V[-1], 0.0)
        self.assertLess(self.segment.V[-1], 1.0)

    def test_signal_end(self):
        """Test the forward propagation through segments with a signal at the end."""

        N = 100
        for i in range(N):
            self.segment.V[-2] = 1.0
            self.engine.forward()

        self.assertGreater(self.segment.V[0], 0.0)
        self.assertLess(self.segment.V[0], 1.0)

    def test_branching(self):
        """Test adding branching segments and ensuring the engine handles it."""

        # Create two new branches
        branch_L = DendriteSegment(**dendrite_default_configuration(self.c), name="L")
        branch_R = DendriteSegment(**dendrite_default_configuration(self.c), name="R")
        self.engine.add_segment(branch_L)
        self.engine.add_segment(branch_R)

        self.engine.add_branch(self.segment, branch_L, branch_R)

        # Check that branches have been added to the lists
        self.assertEqual(len(self.engine.branchesM), 1)
        self.assertEqual(len(self.engine.branchesL), 1)
        self.assertEqual(len(self.engine.branchesR), 1)

    def test_logging(self):
        """Test logging a segment's data to TensorBoard (mock test)."""

        # Mock the TensorBoard writer
        mock_writer = unittest.mock.Mock()

        # Log data
        self.engine.log_to_tensorboard(mock_writer, step=1)

        # Verify the writer was called for the segment's logging
        mock_writer.add_scalars.assert_called()


if __name__ == "__main__":
    unittest.main()
