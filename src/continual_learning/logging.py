from _utils import convert_to_json_serializable

from collections import defaultdict
import numpy as np
import json
import os


from typing import Dict, Optional, Union


class ContinualTrainerLogger:
    """
    A logger class for handling training and validation metrics in continual learning.

    This class manages the storage and logging of metrics during training and validation. It is relevant
    as it is the main source to track the aggregated continual learning metrics based on a model.
    """

    def __init__(
        self,
        experiment_dir: str,
        experiment_name: str,
    ):
        """
        Initialize the TrainerLogger with empty training and validation histories.
        """
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.save_dir = self._create_save_dir()

        # This object captures the metrics for distinct training runs.
        # Indexed by the `task_id` we are training on this run.
        self.training_runs: Dict[str, dict] = {}
        self._training_history = defaultdict(list)
        self._validation_history = defaultdict(lambda: defaultdict(list))

    def _create_save_dir(self) -> str:
        """
        Create and return the save directory path for the current experiment.

        Returns:
            str: Path to the save directory.
        """
        save_dir = os.path.join(
            self.experiment_dir,
            "models",
            self.experiment_name,
        )
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def complete_task(self, task_id: str):
        """
        This method 'closes' the previous trainer task.
        """

        # Store the current training and validation under the task
        self.training_runs[task_id] = {
            "training": self._training_history,
            "validation": self._validation_history,
        }

        # Reset the training and validation history
        self._training_history = defaultdict(list)
        self._validation_history = defaultdict(lambda: defaultdict(list))

    def log_training_metrics(
        self,
        losses: np.ndarray,
        metrics: Dict[str, np.ndarray],
        print_metrics: bool = True,
    ):
        """
        Log training metrics for an epoch.

        ### Args:
            `losses (np.ndarray)`: Array of losses for the epoch.
            `metrics (Dict[str, np.ndarray])`: Dictionary of metrics for the epoch.
        """

        # Store the training history
        self._training_history["loss"].append(losses.mean())
        for metric, values in metrics.items():
            self._training_history[metric].append(values.mean())

        # Log the results
        if print_metrics:
            self.print_logged_metrics(losses, metrics)

    def log_validation_metrics(
        self,
        task_id: str,
        metrics: Dict[str, np.ndarray],
    ):
        """
        Log validation metrics for a specific task.

        ### Args:
            `task_id (str)`: Identifier for the task.
            `losses (np.ndarray)`: Array of losses for the task.
            `metrics (Dict[str, np.ndarray])`: Dictionary of metrics for the task.
        """

        # Update the task history
        for metric, values in metrics.items():
            self._validation_history[task_id][metric].append(values.mean())

    def print_logged_metrics(
        self,
        losses: np.ndarray,
        metrics: Dict[str, np.ndarray],
        task_id: Optional[str] = None,
    ):
        """
        Print the logged metrics for training or validation.

        ### Args:
            `losses (np.ndarray)`: Array of losses.
            `metrics (Dict[str, np.ndarray])`: Dictionary of metrics.
            `task_id (Optional[str])`: Identifier for the task (None for training).
        """

        phase = "Validation" if task_id else "Training"
        task_str = f" for task {task_id}" if task_id else ""
        print(f"\n{phase} metrics{task_str}:")
        print(f"\tLoss: {losses.mean():.4f} ± {losses.std():.4f}")
        for metric, values in metrics.items():
            print(f"\t{metric}: {values.mean():.4f} ± {values.std():.4f}")

    def store_settings(self, settings: Dict[str, Union[str, int]]) -> None:
        """
        Store additional settings in a config file within the save directory.

        Args:
            settings (Dict[str, Union[str, int]]): Dictionary containing additional settings.
        """
        config_path = os.path.join(self.save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(settings, f, indent=4)

    def store_run(self) -> str:
        """
        Store the training and validation history to a JSON file within a structured directory.

        The history is saved under `results/{experiment_name}-{timestamp}/history.json` to maintain
        organized and isolated experiment records. The method ensures that all directories are created
        if they do not exist and that all data is properly serialized for JSON storage.
        """

        # Define the file path for the history JSON
        history_filepath = os.path.join(self.save_dir, "history.json")

        # Convert numpy arrays and floats to lists for JSON serialization
        serializable_history = convert_to_json_serializable(self.training_runs)

        # Save the history to the JSON file
        with open(history_filepath, "w") as f:
            json.dump(serializable_history, f, indent=4)

        return history_filepath
