from _constants import (
    DATASET_MEAN_STD,
    DATASET_MAPPING,
    BASE_DATASET_DIR,
)

from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms, datasets
import torch
import os

from typing import Tuple, List


def split_randomly(
    array: List[int],
    num_splits: int,
) -> List[List[int]]:
    """
    Splits dataset indices into `num_splits` parts randomly with reproducibility.

    Args:
        array (List[int]): The array to split.
        num_splits (int): Number of splits/tasks.
        seed (int): Seed for random operations.

    Returns:
        List[List[int]]: A list containing lists of indices for each split.
    """
    permuted_indices = torch.randperm(len(array)).tolist()
    permuted_array = [array[idx] for idx in permuted_indices]
    k, m = divmod(len(array), num_splits)
    return [
        permuted_array[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(num_splits)
    ]


def get_transformation(
    mean: List[float],
    std: List[float],
    resolution: Tuple[int, int],
    augment: bool = False,
    normalise: bool = True,
) -> Dataset:
    """
    Prepare the dataset by applying the augmentation policy.

    Args:
    input_resolution (int): The resolution to resize the images to.
    mean (List[float]): The mean of the dataset.
    std (List[float]): The standard deviation of the dataset.
    augment (bool): Whether to apply the augmentation policy.

    Returns:
    transforms.Compose: The transformation to apply to the dataset.
    """

    if augment:
        image_transformation = [
            transforms.RandomResizedCrop(resolution, scale=(0.08, 1)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        image_transformation = [
            transforms.Resize(resolution),
        ]

    # Build additional transformations
    additional_transformations = [transforms.ToTensor()]
    if normalise:
        additional_transformations.append(
            transforms.Normalize(mean=mean, std=std),
        )

    # Concat all the transformations
    image_transformation.extend(additional_transformations)
    return transforms.Compose(image_transformation)


def get_dataset_from_name(
    dataset_name: str,
    evaluation: bool = False,
) -> Tuple[datasets.VisionDataset, List[float], List[float]]:
    """
    Returns the instantiated dataset along with its mean and standard deviation.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        Tuple[datasets.VisionDataset, List[float], List[float]]:
            - Instantiated dataset object.
            - Mean values for normalization.
            - Standard deviation values for normalization.

    Raises:
        ValueError: If an unsupported dataset name is provided or dataset constants are not defined.
    """

    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower not in DATASET_MAPPING:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if dataset_name_lower not in DATASET_MEAN_STD:
        raise ValueError(f"Dataset mean and std not defined for: {dataset_name}")

    dataset_class: Dataset = DATASET_MAPPING[dataset_name_lower]
    data_dir = os.path.join(BASE_DATASET_DIR, dataset_name_lower)

    # Start of Selection
    if evaluation:
        train_dataset = dataset_class(root=data_dir, train=True, download=True)
        val_dataset = dataset_class(root=data_dir, train=False, download=True)
        dataset = ConcatDataset([train_dataset, val_dataset])
    else:
        dataset = dataset_class(root=data_dir, train=True, download=True)

    mean = DATASET_MEAN_STD[dataset_name_lower]["mean"]
    std = DATASET_MEAN_STD[dataset_name_lower]["std"]

    return dataset, mean, std
