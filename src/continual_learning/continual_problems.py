"""
Each one of these `ContinualTrainers` should roughly correspond to a different
implementation of a `ContinualDataset` (defined in `continual_datasets.py`).
"""

from continual_learning.continual_datasets import DataContinualDataset
from continual_learning.base_trainer import ContinualModelTrainer

from torch import nn


class DataContinualTrainer(ContinualModelTrainer):
    """
    This class represents a data-continual problem where the number of classes does not
    change across tasks. Can also be referred to a base incremental-learning problem.

    In this kind of problems, as the base model does not need to adapt to different number of
    classes, the `base_model` is not re-organised for the new task-specific head.
    """

    data_class = DataContinualDataset

    def model_task_change(self, task_id: str, task_model: nn.Module):
        """
        No changes are made to the model when a new task is encountered.
        """
        self.base_model = task_model
