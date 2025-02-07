import numpy as np
import random
import torch
from typing import Union, List, Dict


def set_seed(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility
    across different components of the pipeline. This includes both CPU and GPU seeds
    (if CUDA is available).

    Args:
        seed (int): The seed value to use for all random number generators.

    Notes:
        - The PyTorch setting `torch.backends.cudnn.deterministic=True` ensures that
          CUDA operations are deterministic, but this may reduce performance.
        - Setting `torch.backends.cudnn.benchmark=False` ensures that PyTorch does not
          attempt to find the best algorithms for the hardware, which could lead to
          non-deterministic behavior. This may also impact performance slightly.
    """
    # Seed for the Python built-in random library
    random.seed(seed)

    # Seed for NumPy random number generator
    np.random.seed(seed)

    # Seed for PyTorch CPU RNG
    torch.manual_seed(seed)

    # If CUDA is available, seed all CUDA-enabled GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch (optional for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_json_serializable(obj) -> Union[List, Dict, int, float, str]:
    """
    Convert numpy arrays, numpy scalar types, and nested structures to JSON serializable format.

    Args:
        obj: The object to be converted.

    Returns:
        Union[List, Dict, int, float, str]: The converted object in JSON serializable format.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)

    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)

    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}

    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(v) for v in obj]

    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    else:
        print(obj)
        return str(obj)
