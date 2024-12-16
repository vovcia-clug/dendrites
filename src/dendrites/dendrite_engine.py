import torch
from config import MainConfig

from dendrites.boundary.boundary_context import BoundaryContext
from dendrites.forward.forward_context import ForwardContext
from dendrites.segment import DendriteSegment
from dendrites.voltage_cache_table import VoltageCacheTable

EXTEND_STEP = 128


class DendriteEngine:
    def __init__(self,
                 c: MainConfig,
                 forward_context: ForwardContext = None,
                 boundary_context: BoundaryContext = None):
        self.c = c
        self.segments = set()
        self.count = 0
        self.voltage_tables = {}
        self.voltage_cache_data = {}
        self.slice_ids = {}
        self.branchesM = []
        self.branchesL = []
        self.branchesR = []
        self.forward_context = forward_context if forward_context is not None else ForwardContext(c)
        self.boundary_context = boundary_context if boundary_context is not None else BoundaryContext()

    def create_segment(self, *args, **kwargs) -> DendriteSegment:
        segment = DendriteSegment(*args, **kwargs)
        self.add_segment(segment)
        return segment

    def add_segment(self, segment: DendriteSegment):
        LEN = segment.length
        slice_id = self.reserve_slice(segment, LEN)
        self.slice_ids[segment] = slice_id
        self.segments.add(segment)

    def reserve_slice(self, segment: DendriteSegment, LEN):
        if LEN not in self.voltage_tables:
            cache_table = VoltageCacheTable(LEN, EXTEND_STEP)
            self.voltage_tables[LEN] = cache_table
            self.voltage_cache_data[LEN] = cache_table.data
        if not self.voltage_tables[LEN].free:
            self.voltage_tables[LEN].extend(EXTEND_STEP)
            self.voltage_cache_data[LEN] = self.voltage_tables[LEN].data
        slice_id = self.voltage_tables[LEN].reserve_slice(segment)
        self.forward_context.strategy.cache_clear()
        self.count += 1
        return slice_id

    def free_slice(self, slice_id, LEN):
        self.voltage_tables[LEN].free_slice(slice_id)
        if len(self.voltage_tables[LEN].reserved) == 0:
            del self.voltage_tables[LEN]

    def grow(self, segment: DendriteSegment):
        old_length = segment.length
        new_length = segment.length + 1
        old_slice_id = self.slice_ids[segment]
        new_slice_id = self.reserve_slice(segment, new_length)
        self.voltage_tables[new_length].data[new_slice_id, 0, :old_length] = self.voltage_cache_data[old_length][
            old_slice_id, 0]
        self.voltage_tables[new_length].data[new_slice_id, 0, -1] = self.voltage_cache_data[new_length][
            new_slice_id, 0, -2]
        self.free_slice(old_slice_id, segment.length)
        self.slice_ids[segment] = new_slice_id
        segment.set_length(new_length)
        return new_length

    def add_branch(self, segment: DendriteSegment, segment_L: DendriteSegment, segment_R: DendriteSegment):
        self.branchesM.append(segment.V[-3:-1])
        self.branchesL.append(segment_L.V[1:3])
        self.branchesR.append(segment_R.V[1:3])

    def forward(self):
        self._forward_core()
        self._forward_boundary()
        self._forward_branch()

    def _forward_boundary(self):
        self.boundary_context.boundary_strategy.boundary(self.voltage_cache_data)

    def _forward_branch(self):
        if self.branchesM and self.branchesL and self.branchesR:
            self.boundary_context.boundary_strategy.boundary_branch(self.branchesM, self.branchesL, self.branchesR)

    def _forward_core(self):
        self.forward_context.forward(self.voltage_tables)

    def log_to_tensorboard(self, writer, step):
        for segment in self.segments:
            segment.log_to_tensorboard(writer, step)
