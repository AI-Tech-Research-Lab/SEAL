from search_space import get_search_space

import numpy as np
import json
import os

from typing import Dict, Any


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents as a dictionary.

    ### Args:
        `file_path (str)`: Path to the JSON file.

    ### Returns:
        `Dict[str, Any]`: Contents of the JSON file.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def compute_training_metrics(history: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes all continual learning metrics and returns them in a dictionary. All the metrics
    are computed for each task_k.

    ### Parameters:
        `history (Dict[str, Any])`: The training history containing validation accuracies.

    """
    tasks_data = history["training_metrics"]["validation"]

    # Get the accuracies and the flatness
    accuracies = [task_data["accuracy"][-1] for task_data in tasks_data.values()]
    flatness = [task_data["flatness"][-1] for task_data in tasks_data.values()]

    # Get the average accuracy and the flatness at the last task
    average_accuracy = np.mean(accuracies)
    average_flatness = np.mean(flatness)

    return {
        "average_accuracy": average_accuracy,
        "average_flatness": average_flatness,
    }


def postprocess_experiment(
    experiment_dir: str,
    search_space_family: str,
    fixed_arch: bool = False,
) -> Dict[str, Any]:
    """
    Process a single experiment directory and extract relevant information.

    ### Args:
        `experiment_dir (str)`: Path to the experiment directory.

    ### Returns:
        Dict[str, Any]: Dictionary containing extracted information.
    """
    config_path = os.path.join(experiment_dir, "config.json")
    history_path = os.path.join(experiment_dir, "history.json")

    # Get the raw data
    config = load_json(config_path)
    history = load_json(history_path)

    # Get model sample definition
    search_space = get_search_space(search_space_family, fixed=fixed_arch)
    encoded_sample = search_space.encode(config["sample_definition"])

    # Compute the continual metrics from the history
    training_metrics = compute_training_metrics(history)

    # Extract relevant information from config and history
    sample_result = {
        "model": "".join(map(str, encoded_sample)),
        **{k: v for k, v in config.items() if k not in ["sample_definition"]},
        **training_metrics,
    }

    return sample_result
