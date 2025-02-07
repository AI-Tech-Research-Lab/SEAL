from continual_learning.base_dataset import ContinualDataset
from tools.data_utils import split_randomly

from torch.utils.data import Subset

from typing import Dict, List


ConfigType = Dict[str, str | float | int | bool]


class DataContinualDataset(ContinualDataset):
    """
    Implements Data Continual Learning by splitting the dataset into sequential data subsets.
    Each subset represents a different task.
    """

    def split_dataset(self, num_tasks: int):
        dataset_samples = list(range(len(self.dataset)))
        split_indices = split_randomly(dataset_samples, num_tasks)
        return {
            tid: Subset(self.dataset, indices)
            for tid, indices in enumerate(split_indices)
        }


class ClassContinualDataset(ContinualDataset):
    """
    Implements Class Incremental Learning by splitting the dataset based on classes.
    Each subset contains data for a specific set of classes representing a task.
    """

    def split_dataset(self, num_tasks: int):
        """
        Splits the dataset into n subsets based on classes.
        """
        classes = self._get_classes()
        task_classes = split_randomly(classes, num_tasks)

        subsets = {}
        for i, task_cls in enumerate(task_classes):
            indices = self._get_indices_for_classes(task_cls)
            subsets[i] = Subset(self.dataset, indices)
        return subsets

    def _get_classes(self) -> List[int]:
        """
        Retrieves the list of unique classes in the dataset.
        Assumes that dataset.targets exists and is a list or similar.
        """
        if hasattr(self.dataset, "targets"):
            return list(sorted(set(self.dataset.targets)))
        else:
            raise AttributeError("Dataset does not have 'targets' attribute.")

    def _get_indices_for_classes(self, classes: List[int]) -> List[int]:
        """
        Retrieves indices for data points belonging to the specified classes.
        """
        target_list = self.dataset.targets
        return [i for i, target in enumerate(target_list) if target in classes]
