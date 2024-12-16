import torch

from dendrites.exceptions import DendriteCacheFull
from dendrites.segment import DendriteSegment


class VoltageCacheTable:
    def __init__(self, length: int, initial_size: int):
        self.length = length
        self.reserved = dict()
        self.data = torch.zeros((initial_size, 2, length), dtype=torch.float)
        self._free = set(range(initial_size))

    @property
    def total(self):
        return self.data.shape[0]

    @property
    def free(self):
        return len(self._free)

    def extend(self, num: int):
        self._free.update(range(self.total, self.total + num))
        self.data = torch.cat((self.data, torch.zeros((num, 2, self.length), dtype=torch.float)), dim=0)
        for slice_index in self.reserved:
            self.set_dendrite(slice_index)

    def reserve_slice(self, dendrite: DendriteSegment):
        try:
            slice_index = self._free.pop()
        except KeyError:
            raise DendriteCacheFull
        self.reserved[slice_index] = dendrite
        self.set_dendrite(slice_index)
        return slice_index

    def free_slice(self, slice_index):
        del self.reserved[slice_index]
        self._free.add(slice_index)
        return slice_index

    def set_dendrite(self, slice_index):
        self.reserved[slice_index].V = self.data[slice_index, 0]
        self.data[slice_index, 1] = self.reserved[slice_index].D_batch(self.length)
