# ruff: noqa: F401
import subprocess
import os

from typing import List, Tuple

FIXED_MODEL_SCRIPT_FILEPATH = "src/nas/evaluators/fixed_evaluator.py"
GROWING_MODEL_SCRIPT_FILEPATH = "src/nas/evaluators/growing_evaluator.py"


def _evaluate_model(
    script_filepath: str,
    encoded_sample: List[int],
    random_seed: int,
    experiment_dir: str,
    ofa_space_family: str,
    dataset_name: str,
    n_tasks: int,
    optimiser_name: str,
    learning_rate: float,
    weight_decay: float,
    epochs_per_task: int,
):
    """
    Evaluate a single model with the given script.

    Args:
        encoded_sample (List[int]): The encoded model sample to evaluate.
        seed (int): Random seed for reproducibility.
        experiment_dir (str): Directory to store experiment results.
        ofa_space_family (str): OFA space family.
        dataset_name (str): Name of the dataset.
        n_tasks (int): Number of tasks.
        optimiser_name (str): Name of the optimizer.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay.
        epochs_per_task (int): Number of epochs per task.

    Returns:
        Tuple[str, str]: A tuple containing the encoded sample and the experiment directory.
    """

    # Run the trainer for the model
    cmd = [
        "python",
        script_filepath,
        "--experiment_dir",
        str(experiment_dir),
        "--model_encoding",
        *map(str, encoded_sample),
        "--search_space_family",
        str(ofa_space_family),
        "--dataset",
        str(dataset_name),
        "--n_tasks",
        str(n_tasks),
        "--optimiser_name",
        str(optimiser_name),
        "--learning_rate",
        str(learning_rate),
        "--weight_decay",
        str(weight_decay),
        "--epochs_per_task",
        str(epochs_per_task),
        "--random_seed",
        str(random_seed),
    ]
    subprocess.run(cmd, check=True)

    # Get the history file path stored by the trainer
    results_dir = os.path.join(
        experiment_dir,
        "models",
        f'{"".join(map(str, encoded_sample))}-{random_seed}',
    )
    return (encoded_sample, results_dir)


def evaluate_fixed_model(
    encoded_sample: List[int],
    random_seed: int,
    experiment_dir: str,
    ofa_space_family: str,
    dataset_name: str,
    n_tasks: int,
    optimiser_name: str,
    learning_rate: float,
    weight_decay: float,
    epochs_per_task: int,
    capacity_tau: float,
    expand_is_frozen: bool,
    distill_on_expand: bool,
    weights_from_ofa: bool,
):
    """
    Evaluate a single model with the fixed architecture script.

    Args:
        seed (int): Random seed for reproducibility.
        experiment_dir (str): Directory to store experiment results.
        ofa_space_family (str): OFA space family.
        dataset_name (str): Name of the dataset.
        n_tasks (int): Number of tasks.
        optimiser_name (str): Name of the optimizer.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay.
        epochs_per_task (int): Number of epochs per task.

    Returns:
        Tuple[str, str]: A tuple containing the encoded sample and the experiment directory.
    """

    from .fixed_evaluator import main

    main(
        experiment_dir,
        encoded_sample,
        ofa_space_family,
        dataset_name,
        n_tasks,
        optimiser_name,
        learning_rate,
        weight_decay,
        epochs_per_task,
        random_seed,
    )

    # Get the history file path stored by the trainer
    results_dir = os.path.join(
        experiment_dir,
        "models",
        f'{"".join(map(str, encoded_sample))}-{random_seed}',
    )
    return (encoded_sample, results_dir)


def evaluate_growing_model(
    encoded_sample: List[int],
    random_seed: int,
    experiment_dir: str,
    ofa_space_family: str,
    dataset_name: str,
    n_tasks: int,
    optimiser_name: str,
    learning_rate: float,
    weight_decay: float,
    epochs_per_task: int,
    capacity_tau: float,
    expand_is_frozen: bool,
    distill_on_expand: bool,
    weights_from_ofa: bool,
) -> Tuple[List[int], str]:
    """
    Evaluate a single model with the growing architecture script.

    Args:
        seed (int): Random seed for reproducibility.
        experiment_dir (str): Directory to store experiment results.
        ofa_space_family (str): OFA space family.
        dataset_name (str): Name of the dataset.
        n_tasks (int): Number of tasks.
        optimiser_name (str): Name of the optimizer.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay.
        epochs_per_task (int): Number of epochs per task.

    Returns:
        Tuple[str, str]: A tuple containing the encoded sample and the experiment directory.
    """

    from .growing_evaluator import main

    main(
        experiment_dir,
        encoded_sample,
        ofa_space_family,
        dataset_name,
        n_tasks,
        optimiser_name,
        learning_rate,
        weight_decay,
        epochs_per_task,
        capacity_tau,
        expand_is_frozen,
        distill_on_expand,
        weights_from_ofa,
        random_seed,
    )

    # Get the history file path stored by the trainer
    results_dir = os.path.join(
        experiment_dir,
        "models",
        f'{"".join(map(str, encoded_sample))}-{random_seed}',
    )
    return (encoded_sample, results_dir)
