from tools.data_utils import get_transformation

from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod

from typing import Dict, List


ConfigType = Dict[str, str | float | int | bool]


class DataSubset(Dataset):
    """
    A subset of a continual dataset with an applied transformation.

    Args:
        subset (Subset): The original subset of the dataset.
        transform (Callable): The transformation to apply to the data.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class ContinualDataset(ABC):
    """
    Abstract class for a continual learning dataset.
    """

    def __init__(self, dataset: Dataset, num_tasks: int, normalise: bool = True):
        self.dataset = dataset

        # This are defined in the class specific dataset
        self.normalise = normalise
        self.DATA_MEAN = []
        self.DATA_STD = []

        # Splits the given dataset according to the strategy defined by the class.
        self.num_tasks = num_tasks
        self.tasks: Dict[str, Dataset] = self.split_dataset(num_tasks)

    @abstractmethod
    def split_dataset(self, num_tasks: int) -> Dict[str, Dataset]:
        pass

    def set_data_params(self, mean: List[float], std: List[float]):
        """
        Sets the data mean and standard deviation for normalization.

        ### Args:
            `mean (List[float])`: The mean of the dataset.
            `std (List[float])`: The standard deviation of the dataset.
        """
        self.DATA_MEAN = mean
        self.DATA_STD = std

    def get_task_loader(
        self,
        task_id: str,
        resolution: int,
        batch_size: int,
        shuffle: bool = False,
        augment: bool = False,
        num_workers: int = 1,
    ) -> DataLoader:
        """
        Returns a DataLoader for the current training split with training transformations.

        Args:
            batch_size (int): The batch size for the DataLoader.
            task_id (str): Identifier of the current task to use for training.
            training_transform (Callable, optional): Transformation to apply to the training data.

        Returns:
            DataLoader: Training DataLoader for the current task.
        """
        assert task_id in self.tasks, f"Task {task_id} not found in dataset."
        training_subset = self.tasks[task_id]

        # Builds the training transformation step from dataset params
        training_transformer = get_transformation(
            mean=self.DATA_MEAN,
            std=self.DATA_STD,
            resolution=resolution,
            augment=augment,
            normalise=self.normalise,
        )

        # Applies the transformation to the training subset
        transformed_training_subset = DataSubset(
            training_subset,
            transform=training_transformer,
        )

        return DataLoader(
            transformed_training_subset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )
