from .base_space import BaseOFASearchSpace

from typing import List


class FixedSearchSpace(BaseOFASearchSpace):
    """
    This class is just wrapper of the base cases to provide a single interface for
    the different definition os Search Spaces.
    """

    def sample(self, n_samples: int = 1):
        return self._base_sample(
            n_samples,
            num_blocks=self.num_blocks,
            depths=self.block_depths,
            widths=self.block_widths,
            ksizes=self.block_ksizes,
            resolutions=self.input_resolutions,
        )

    def encode(self, sample):
        return self._base_encode(sample)

    def decode(self, sample: List[int]):
        return self._base_decode(sample)
