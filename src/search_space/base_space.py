from abc import ABC, abstractmethod
import numpy as np

from typing import List, Dict, TypeAlias


ModelSample: TypeAlias = Dict[str, List[int] | int]


class BaseOFASearchSpace(ABC):
    """
    Defines the base OFA net search space.

    Some standard descriptors:
    - `num_blocks`: max number of units in the CNN sequence.
    - `block_depths`: number of layers on each block.
    - `block_widths`: expansion rate for the number of channels on each layer.
    - `block_ksizes`: kernel size of each layer.
    - `input_resolutions`: varies the resolution of the input image
    """

    def __init__(self, family: str, fixed_res: bool = False):
        match family:
            case "mobilenetv3":
                self.num_blocks = 5
                self.block_depths = [2, 3, 4]
                self.block_widths = [3, 4, 6]
                self.block_ksizes = [3, 5, 7]
                self.input_resolutions = list(range(128, 224, 4))
                self.nvar = 45 + (1 if not fixed_res else 0)
            case _:
                raise KeyError(f"OFA family type: '{family}' not implemented!")

    def _zero_padding(self, values: List[int], depths: List[int]) -> List[int]:
        """
        Pads the given values to the max depth available for the OFA model.

        ### Args:
        - `values (List[int])`: A flattened list with the values for each of the layers.
        - `depths (List[int])`: A list with the depth of each block.

        ### Returns:
        - `padded_values (List[int])`: A list with the padded values.

        ### Example:
        ```python
        depths = [2, 3, 1]
        values = [3, 5, 3, 5, 7, 3]
        expected = [3, 5, 0, 0, 3, 5, 7, 0, 3, 0, 0, 0]
        ```
        """
        padded_values, position = [], 0
        for d in depths:
            for _ in range(d):
                padded_values.append(values[position])
                position += 1
            padded_values += [0] * (max(self.block_depths) - d)
        return padded_values

    def _base_sample(
        self,
        n_samples: int,
        num_blocks: int,
        depths: List[int],
        widths: List[int],
        ksizes: List[int],
        resolutions: List[int],
    ) -> List[ModelSample]:
        """
        Encloses the logic to perform a basic random sampling based on the given possible
        values for each parameter.

        #### Args
        - `n_samples (int)`: Number of samples to generate.
        - `num_blocks (int)`: Number of blocks in the network.
        - `depths (List[int])`: List of possible depths for each block.
        - `widths (List[int])`: List of possible widths for each layer.
        - `ksizes (List[int])`: List of possible kernel sizes for each layer.
        - `resolutions (List[int])`: List of possible resolutions for the input image.

        ### Returns
        A list of samples, where each sample is a dictionary with the following structure:
        """

        samples = []
        for _ in range(n_samples):
            sampled_resolution = np.random.choice(resolutions)

            # The depth defines the number of layers we are defining
            sampled_depth = np.random.choice(depths, num_blocks, replace=True)
            n_layers = sampled_depth.sum()

            # Other parameters are extracting depending on n_layers
            sampled_widths = np.random.choice(widths, n_layers, replace=True)
            sampled_ksizes = np.random.choice(ksizes, n_layers, replace=True)

            # Append the sampled architecture
            samples.append(
                {
                    "depths": sampled_depth.tolist(),
                    "ksizes": sampled_ksizes.tolist(),
                    "widths": sampled_widths.tolist(),
                    "resolution": int(sampled_resolution),
                }
            )
        return samples

    def _base_encode(self, sample: ModelSample) -> List[int]:
        """
        Performs the fix-length encoding to a integer string of a sample.
        Made some changes for readibility of the encodings( and to follow the paper example ).

        ### Args
        - `sample (ModelSample)`: A dict of shape

        ```python
        {
            'resolution': int,
            'depths': list(int),
            'ksizes': list(int),
            'widths': list(int)
        }
        ```

        ### Returns
        - A list with "6-blocks" (5 + input).
        - Each block is represented by 9 sequential elements:

        - `block-depth (x1) | padded( ksizes ) (x4) | padded(exp_rates) (x4)`

        The overall structure is:
        `block1 | block2 | block3 | block4 | block5 | input_rest`
        """
        encoding = []

        # Get idx for the sample parameters
        _depths = [
            np.argwhere(_x == np.array(self.block_depths))[0, 0]
            for _x in sample["depths"]
        ]
        _ksizes = [
            np.argwhere(_x == np.array(self.block_ksizes))[0, 0]
            for _x in sample["ksizes"]
        ]
        _widths = [
            np.argwhere(_x == np.array(self.block_widths))[0, 0]
            for _x in sample["widths"]
        ]

        # Pad the structures to ensure same length
        ksizes = self._zero_padding(_ksizes, sample["depths"])
        widths = self._zero_padding(_widths, sample["depths"])

        max_depth = max(self.block_depths)
        for i in range(self.num_blocks):
            encoding += [_depths[i]]
            encoding += ksizes[i * max_depth : (i + 1) * max_depth]
            encoding += widths[i * max_depth : (i + 1) * max_depth]

        encoding += [self.input_resolutions.index(sample["resolution"])]
        return encoding

    def _base_decode(self, sample: List[int]) -> ModelSample:
        """
        Performs the opposite operation from encoding. Decodes the sample
        from a representative list of string values.

        ### Args:
        - `sample (List[int])`: Is assumed to be in the form:

        `[...block1, ...block2, ..., ...block5, resolution]`
        """

        # Computes the overall size of each block
        max_depth = max(self.block_depths)
        encoded_block_size = 2 * max_depth + 1

        # Transform to array for easier indexing
        _ksizes = np.array(self.block_ksizes)
        _widths = np.array(self.block_widths)

        # Reconstruct from the index in the sample
        depths, ksizes, widths = [], [], []
        for i in range(0, len(sample) - 1, encoded_block_size):
            n_layers = self.block_depths[sample[i]]

            depths.append(n_layers)
            ksizes.extend(_ksizes[sample[i + 1 : i + 1 + n_layers]].tolist())
            widths.extend(
                _widths[
                    sample[i + (1 + max_depth) : i + (1 + max_depth) + n_layers]
                ].tolist()
            )

        # Reconstruct the resolution from the last element
        _resolution_index = sample[-1]
        _resolution = self.input_resolutions[_resolution_index]

        return {
            "depths": depths,
            "ksizes": ksizes,
            "widths": widths,
            "resolution": _resolution,
        }

    def _get_min_sample(self) -> ModelSample:
        """
        Return the smallest possible architecture given the OFA family.
        """
        return self._base_sample(
            n_samples=1,
            num_blocks=self.num_blocks,
            depths=[min(self.block_depths)],
            widths=[min(self.block_widths)],
            ksizes=[min(self.block_ksizes)],
            resolutions=[min(self.input_resolutions)],
        )[0]

    def _get_max_sample(self) -> ModelSample:
        """
        Return the largest possible architecture given the OFA family.
        """
        return self._base_sample(
            n_samples=1,
            num_blocks=self.num_blocks,
            depths=[max(self.block_depths)],
            widths=[max(self.block_widths)],
            ksizes=[max(self.block_ksizes)],
            resolutions=[max(self.input_resolutions)],
        )[0]

    def get_initial_samples(self, n_archs: int) -> List[ModelSample]:
        """
        Returns a list of initial samples to start the search.
        """
        data = [self._get_min_sample(), self._get_max_sample()]
        data.extend(self.sample(n_samples=n_archs - 2))
        return data

    @abstractmethod
    def encode(self, sample: ModelSample) -> List[int]:
        """
        Defines the final interface for encoding a single sample
        """
        pass

    @abstractmethod
    def decode(self, sample: List[int]) -> ModelSample:
        """
        Defines the final interface for decoding a single sample
        """
        pass

    @abstractmethod
    def sample(self, n_samples: int = 1, **kwargs) -> List[ModelSample]:
        """
        Defines the final interface for sampling multiple samples.
        """
        pass
