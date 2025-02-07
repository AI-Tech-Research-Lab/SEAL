from .base_space import BaseOFASearchSpace, ModelSample

import numpy as np

from typing import List


class GrowthSearchSpace(BaseOFASearchSpace):
    """
    Implements the Search Space with the growth direction encoded in the search space.
    """

    def __init__(self, family: str, expand_from_end: bool = True):
        super().__init__(family, fixed_res=False)

        # Define the number growth direction bits
        self.n_grow_bits = 3
        self.nvar = self.nvar + self.n_grow_bits

        # Expand from the begining or the end of the model
        self.expand_from_end = expand_from_end

    def _get_min_sample(self) -> ModelSample:
        """
        We need to overwrite this method to account for the constrained space.
        """
        _sample = self._base_sample(
            n_samples=1,
            num_blocks=self.num_blocks,
            depths=[min(self.block_depths)],
            widths=[min(self.block_widths)],
            ksizes=[min(self.block_ksizes)],
            resolutions=[min(self.input_resolutions)],
        )[0]

        _direction = np.zeros(self.n_grow_bits, dtype=int)
        return {**_sample, "direction": _direction.tolist()}

    def _get_max_sample(self) -> ModelSample:
        """
        We need to overwrite this method to account for the constrained space.
        """
        _sample = self._base_sample(
            n_samples=1,
            num_blocks=self.num_blocks,
            depths=[max(self.block_depths[:-1])],
            widths=[max(self.block_widths[:-1])],
            ksizes=[max(self.block_ksizes[:-1])],
            resolutions=[max(self.input_resolutions)],
        )[0]

        _direction = np.ones(self.n_grow_bits, dtype=int)
        return {**_sample, "direction": _direction.tolist()}

    def encode(self, sample: ModelSample):
        """
        Defines the encoding of the architecture. The model part is the same as before,
        but we add the direction bits ( they are always a bit List ).
        """
        return self._base_encode(sample) + sample["direction"]

    def decode(self, sample: List[int]) -> ModelSample:
        """
        Decodes the sample into a ModelSample.

        Note: to use the `self._base_decode` method, the last bit of the passed encoding
        should be the resolution.
        """
        encoded_base_model = sample[: -self.n_grow_bits]
        return {
            **self._base_decode(encoded_base_model),
            "direction": [int(i) for i in sample[-self.n_grow_bits :]],
        }

    def sample(self, n_samples: int = 1) -> List[ModelSample]:
        """
        Sampling works a bit different as we have to make sure we have a valid space on which
        to perform the model growth.
        """

        # Sample the base architectures under a constraint space.
        base_samples = self._base_sample(
            n_samples,
            num_blocks=self.num_blocks,
            depths=self.block_depths[:-1],
            widths=self.block_widths[:-1],
            ksizes=self.block_ksizes[:-1],
            resolutions=self.input_resolutions,
        )

        # Add the scaling direction to each sample
        samples_with_growth = []
        for sample in base_samples:
            direction = np.random.choice([0, 1], size=self.n_grow_bits, replace=True)
            sample["direction"] = direction.tolist()
            samples_with_growth.append(sample)

        return samples_with_growth

    def _apply_depth_scaling(
        self,
        scaled_ksizes: List[int],
        scaled_widths: List[int],
        scaled_depths: List[int],
    ) -> tuple[list, list, list]:
        """
        Applies depth scaling to the first available block.

        Args:
            base_sample (ModelSample): The base architecture sample

        Returns:
            tuple[list, list, list]: Updated depths, ksizes, and widths lists
        """

        # The widths are traversed in reverse order
        reversed_depths = scaled_depths[::-1] if self.expand_from_end else scaled_depths

        # Define the positions of the corresponding layers
        _pos = sum(scaled_depths)

        for _bid, block_depth in enumerate(reversed_depths):
            bid = len(reversed_depths) - _bid - 1 if self.expand_from_end else _bid
            _pos -= block_depth

            # Expand the first found depth that can be expanded
            if block_depth < max(self.block_depths):
                # Select the current depth index
                next_depth_idx = self.block_depths.index(block_depth) + 1
                scaled_depths[bid] = self.block_depths[next_depth_idx]

                # Update ksizes and widths for the new layer
                block_ksizes = scaled_ksizes[_pos : _pos + block_depth]
                block_widths = scaled_widths[_pos : _pos + block_depth]

                # Copy the last layer before scaling
                insert_pos = _pos + scaled_depths[bid] - 1
                scaled_ksizes.insert(insert_pos, block_ksizes[-1])
                scaled_widths.insert(insert_pos, block_widths[-1])
                break

        return scaled_depths, scaled_ksizes, scaled_widths

    def _apply_width_scaling(self, widths: list) -> list:
        """
        Applies width scaling to the first available layer.

        Args:
            widths (list): Current width values

        Returns:
            list: Updated width values
        """
        scaled_widths = widths.copy()

        # We traverse the widths in reverse order ( expand from the end )
        reversed_widths = scaled_widths[::-1] if self.expand_from_end else scaled_widths

        for _wid, block_width in enumerate(reversed_widths):
            wid = len(reversed_widths) - _wid - 1 if self.expand_from_end else _wid

            # Expand the first found width that can be expanded
            if block_width < max(self.block_widths):
                _width_idx = self.block_widths.index(block_width) + 1
                scaled_widths[wid] = self.block_widths[_width_idx]
                break

        return scaled_widths

    def _apply_ksize_scaling(self, ksizes: list) -> list:
        """
        Applies kernel size scaling to the first available layer.

        Args:
            ksizes (list): Current kernel size values

        Returns:
            list: Updated kernel size values
        """

        scaled_ksizes = ksizes.copy()

        # We traverse the ksizes in reverse order ( expand from the end )
        reversed_ksizes = scaled_ksizes[::-1] if self.expand_from_end else scaled_ksizes

        for _kid, block_ksize in enumerate(reversed_ksizes):
            kid = len(reversed_ksizes) - _kid - 1 if self.expand_from_end else _kid

            # Expand the first found ksize that can be expanded
            if block_ksize < max(self.block_ksizes):
                _ksize_idx = self.block_ksizes.index(block_ksize) + 1
                scaled_ksizes[kid] = self.block_ksizes[_ksize_idx]
                break

        return scaled_ksizes

    def apply_scaling(
        self,
        base_sample: ModelSample,
        direction: List[int],
    ) -> ModelSample:
        """
        Applies the scaling direction to the base model architecture.

        Args:
            base_sample (ModelSample): The base sample to apply the scaling to
            direction (List[int]): The direction to apply [depth, width, ksize]

        Returns:
            ModelSample: The scaled model sample
        """
        scaled_depths = base_sample["depths"].copy()
        scaled_ksizes = base_sample["ksizes"].copy()
        scaled_widths = base_sample["widths"].copy()

        ddir, wdir, kdir = direction

        # Apply the scaling in order for the model expansion
        if kdir > 0:
            scaled_ksizes = self._apply_ksize_scaling(scaled_ksizes)

        if wdir > 0:
            scaled_widths = self._apply_width_scaling(scaled_widths)

        if ddir > 0:
            scaled_depths, scaled_ksizes, scaled_widths = self._apply_depth_scaling(
                scaled_ksizes,
                scaled_widths,
                scaled_depths,
            )

        return {
            "depths": scaled_depths,
            "ksizes": scaled_ksizes,
            "widths": scaled_widths,
            "resolution": base_sample["resolution"],
        }
