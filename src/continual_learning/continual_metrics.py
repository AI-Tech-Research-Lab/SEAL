# This file defines the basic metrics on which to evaluate continual learning models.
# These are evaluated post-training, usually based on accuracy of validation set.
import numpy as np

from typing import Dict, Any, Tuple


def average_accuracy_at_task(accuracies_matrix: np.ndarray, task_k: int) -> float:
    """
    Calculates the average accuracy at a specific task.
    Note that this evaluates only the tasks seen until the `task_id`.

    > For `task_k=1`, we will only see the average acc of `tid=1`.

    ### Args:
        `accuracies_matrix (np.ndarray)`: The TxT matrix of continual learning accuracies.
        `task_k (int)`: The task ID for which to calculate the average accuracy (1-indexed).

    ### Returns:
        `float`: The average accuracy at the specified task.
    """
    if task_k < 1 or task_k > accuracies_matrix.shape[0]:
        raise ValueError(f"Task ID must be between 1 and {accuracies_matrix.shape[0]}")

    # Convert to 0-indexed
    k = task_k - 1

    # Calculate the average accuracy for the specified task
    average_accuracy = np.mean(accuracies_matrix[k, : k + 1])
    return float(average_accuracy)


def average_incremental_accuracy_at_task(
    accuracies_matrix: np.ndarray,
    task_k: int,
) -> float:
    """
    Calculates the average incremental accuracy at a specific task.

    This function computes the average of all average accuracies up to and including
    the specified task.

    ### Args:
        `accuracies_matrix (np.ndarray)`: The TxT matrix of continual learning accuracies.
        `task_k (int)`: The task ID for which to calculate the average incremental accuracy (1-indexed).

    ### Returns:
        `float`: The average incremental accuracy at the specified task.

    Raises:
        ValueError: If task_id is out of bounds.
    """
    if task_k < 1 or task_k > accuracies_matrix.shape[0]:
        raise ValueError(f"Task ID must be between 1 and {accuracies_matrix.shape[0]}")

    # Calculate average accuracies for all tasks up to task_id
    tasks_indices = np.arange(1, task_k + 1)
    average_accuracies = np.array(
        [average_accuracy_at_task(accuracies_matrix, i) for i in tasks_indices],
        dtype=np.float32,
    )

    # Calculate the average of these average accuracies
    return float(np.mean(average_accuracies))


def forgetting_measure_of_task_at_task(
    accuracies_matrix: np.ndarray,
    task_j: int,
    task_k: int,
) -> float:
    """
    Calculates the forgetting measure for task j after training up to task k.

    The forgetting measure is defined as the difference between the maximum accuracy
    achieved on task j during training up to task k-1 and the accuracy on task j
    after training on task k.

    Note: here, taking the max helps to take into account knowledge on the task
    gained by learning other tasks. Explained in https://arxiv.org/pdf/1801.10112.

    ### Args:
        `tasks_matrix (np.ndarray)`: The TxT matrix of continual learning accuracies.
        `task_j (int)`: The task for which to calculate the forgetting measure (1-indexed).
        `task_k (int)`: The current task up to which the model has been trained (1-indexed).

    ### Returns:
        `float`: The forgetting measure for task j after training up to task k.

    ### Raises:
        `ValueError`: If task_j or task_k are out of bounds or if task_j >= task_k.
    """
    if (
        task_j < 1
        or task_k < 1
        or task_j >= task_k
        or task_k > accuracies_matrix.shape[0]
    ):
        raise ValueError(
            f"Invalid task indices. Ensure 1 <= task_j < task_k <= {accuracies_matrix.shape[0]}"
        )

    # Convert to 0-indexed
    j, k = task_j - 1, task_k - 1

    # Compute the diff between max and current.
    max_accuracy = np.max(accuracies_matrix[:k, j])
    current_accuracy = accuracies_matrix[k, j]
    return float(max_accuracy - current_accuracy)


def average_forgetting_at_task(
    accuracies_matrix: np.ndarray,
    task_k: int,
) -> float:
    """
    Calculates the average forgetting measure at task k.

    This function computes the average forgetting measure for all tasks seen
    up to task k-1, normalized by the number of tasks seen previously.

    Returns np.nan for task_k < 2 as forgetting is not meaningful for the first task.

    ### Args:
        `accuracies_matrix (np.ndarray)`: The TxT matrix of continual learning accuracies.
        `task_k (int)`: The current task up to which the model has been trained (1-indexed).

    ### Returns:
        `float`: The average forgetting measure at task k, or np.nan if task_k < 2.
    """
    if task_k > accuracies_matrix.shape[0]:
        raise ValueError(f"Task k must be between 1 and {accuracies_matrix.shape[0]}")

    # Undefined on the first task
    if task_k <= 1:
        return np.nan

    # Calculate forgetting for all previous tasks
    tasks_indices = np.arange(1, task_k)
    forgetting_measures = np.array(
        [
            forgetting_measure_of_task_at_task(accuracies_matrix, j, task_k)
            for j in tasks_indices
        ],
        dtype=np.float32,
    )

    # Calculate the average forgetting
    return float(np.mean(forgetting_measures))


def backward_transfer_at_task(accuracies_matrix: np.ndarray, task_k: int) -> float:
    """
    Calculates the Backward Transfer (BWT) metric up to a specific task.

    Backward Transfer measures the influence that learning tasks up to task_k has on the
    performance of previous tasks. Positive BWT indicates that learning later tasks improved
    performance on earlier tasks, while negative BWT indicates forgetting.

    ### Parameters:
        `accuracies_matrix (np.ndarray)`: The TxT matrix of continual learning accuracies.
        `task_k (int)`: The task up to which to calculate the backward transfer (1-indexed).

    ### Returns:
        `float`: The Backward Transfer score up to task_k, or np.nan if task_k < 2.
    """

    if task_k > accuracies_matrix.shape[0]:
        raise ValueError(f"task_k must be between 1 and {accuracies_matrix.shape[0]}")

    # Undefined on the first task
    if task_k <= 1:
        return np.nan

    # Convert to 0-indexed
    k = task_k - 1

    # Performance of tasks k at tasks k
    task_k_accuracies = np.diag(accuracies_matrix)[:k]

    # Calculate the sum of differences between performance at task k and initial performance
    # for tasks 1 to k-1
    bwt_k = np.mean(accuracies_matrix[k, :k] - task_k_accuracies)
    return float(bwt_k)


def forward_transfer_at_task(
    accuracies_matrix: np.ndarray,
    random_accuracies: np.ndarray,
    task_k: int,
) -> float:
    """
    Calculates the Forward Transfer (FWT) metric up to a specific task.

    Forward Transfer measures the influence that learning previous tasks has on the
    performance of future tasks, compared to the performance of a random initialization.
    Positive FWT indicates that learning earlier tasks improved performance on later tasks.

    ### Args:
        `accuracies_matrix (np.ndarray)`: The TxT matrix of continual learning accuracies.
        `random_accuracies (np.ndarray)`: The vector of random task accuracies (baseline performance).
        `task_k (int)`: The task up to which to calculate the forward transfer (1-indexed).

    ### Returns:
        float: The Forward Transfer score up to task_k, or np.nan if task_k is the final task.
    """
    if task_k > accuracies_matrix.shape[0]:
        raise ValueError(f"task_k must be between 1 and {accuracies_matrix.shape[0]}")

    # Undefined on the first task ( is always 0 )
    if task_k == 1:
        return np.nan

    # Convert to 0-indexed
    k = task_k - 1

    # Get the random accuracy for task k (a*_k)
    random_accuracy = random_accuracies[1 : k + 1]

    # Get the continual learning accuracy for task k before training on task k (a_k-1,k)
    cont_accuracy = accuracies_matrix[np.arange(1, k + 1), np.arange(k)]

    # Calculate the sum of differences between performance before learning task i
    # and the random initialization performance for tasks 2 to k
    fwt_k = np.mean(cont_accuracy - random_accuracy)
    return float(fwt_k)


def get_task_matrices(history: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts task accuracies from the training history.

    Args:
        history (Dict[str, Any]): The training history containing validation accuracies.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            1. A vector of random task accuracies.
            2. A TxT matrix of continual learning accuracies where a[i][j] is the accuracy
               of task j when trained up to task i.

    """
    random_tasks = [tid for tid in history.keys() if tid.startswith("random_")]
    continual_tasks = [tid for tid in history.keys() if tid.startswith("cont_")]

    # This should be the same for both kind of tasks
    n_tasks = len(continual_tasks)

    # Get the random accuracies vector
    random_accuracies = np.array(
        [history[tid]["validation"][tid]["accuracy"][-1] for tid in random_tasks]
    )

    # Get the continual accuracies matrix
    continual_accuracies = np.zeros((n_tasks, n_tasks))
    for i, task in enumerate(continual_tasks):
        task_history = history[task]["validation"]
        for j, eval_task in enumerate(continual_tasks):
            continual_accuracies[i, j] = task_history[eval_task]["accuracy"][-1]

    return (random_accuracies, continual_accuracies)


def compute_continual_metrics(history: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes all continual learning metrics and returns them in a dictionary. All the metrics
    are computed for each task_k.

    ### Parameters:
        `history (Dict[str, Any])`: The training history containing validation accuracies.

    ### Returns:
        `Dict[str, float]`: A dictionary containing all computed metrics with their names as keys.
        ```python
        {
            "average_accuracy"
            "average_incremental_accuracy"
            "average_forgetting"
            "backward_transfer"
            "forward_transfer"
        }
        ```
    """

    # Get the random accuracies vector and the tasks matrix
    random_accuracies, tasks_matrix = get_task_matrices(history)

    # Get the number of tasks
    n_tasks = tasks_matrix.shape[0]

    # We add the forward metric only if we have random accuracies
    learning_metrics = {
        "average_accuracy": [],
        "average_incremental_accuracy": [],
        "average_forgetting": [],
        "backward_transfer": [],
    }

    # Add only if random accuracies are available
    if random_accuracies.shape[0] > 1:
        learning_metrics["forward_transfer"] = []

    # Compute the metrics as a time series
    for task_k in range(1, n_tasks + 1):
        learning_metrics["average_accuracy"].append(
            average_accuracy_at_task(tasks_matrix, task_k=task_k)
        )
        learning_metrics["average_incremental_accuracy"].append(
            average_incremental_accuracy_at_task(tasks_matrix, task_k=task_k)
        )
        learning_metrics["average_forgetting"].append(
            average_forgetting_at_task(tasks_matrix, task_k=task_k)
        )
        learning_metrics["backward_transfer"].append(
            backward_transfer_at_task(tasks_matrix, task_k=task_k)
        )

        # Add only if random accuracies are available
        if "forward_transfer" in learning_metrics:
            learning_metrics["forward_transfer"].append(
                forward_transfer_at_task(tasks_matrix, random_accuracies, task_k=task_k)
            )

    return learning_metrics
