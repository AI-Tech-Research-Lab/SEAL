from custom.sam import SAM

from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from tqdm import tqdm
import numpy as np

from typing import Callable, Dict, Tuple, Iterable, Any


def disable_running_stats(model):
    """
    Used only for SAM optimiser. It disables the running stats like BatchNorm to avoid
    accumulating gradients in the BN layers during the second step.
    """

    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    """
    Used only for SAM optimiser. It enables back the running stats like BatchNorm to be
    able to compute them for the first step.
    """

    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def train_epoch(
    task_model: nn.Module,
    data_loader: DataLoader,
    optimiser: optim.Optimizer,
    loss_fn: Callable,
    metrics: Dict[str, Callable],
    device: torch.device,
    show_progress: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Perform a single training epoch.

    ### Args:
        `model (nn.Module)`: Model to train.
        `data_loader (DataLoader)`: DataLoader for training data.
        `optimiser (optim.Optimizer)`: Optimiser to use for training.
        `loss_fn (Callable)`: Loss function to use for training.
        `metrics (Dict[str, Callable])`: Dictionary of metric functions.
        `device (torch.device)`: Device to perform training on.

    ### Returns:
        `Tuple[np.ndarray, Dict[str, np.ndarray]]`:
            - Array of losses for the epoch.
            - Dictionary of metrics for the epoch.
    """

    task_model.train()

    # Keep track of the losses and metrics
    _losses, _metrics = [], {metric: [] for metric in metrics}

    # Iterate over the dataset
    data_iterator = tqdm(data_loader, desc="Training") if show_progress else data_loader
    for inputs, labels in data_iterator:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # When using SAM, we enable back the running stats.
        if not isinstance(optimiser, SAM):
            optimiser.zero_grad()
        else:
            enable_running_stats(task_model)

        # Standard gradients computation.
        outputs = task_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Optimisation step, in SAM we use the double step.
        if not isinstance(optimiser, SAM):
            optimiser.step()
        else:
            optimiser.first_step(zero_grad=True)
            disable_running_stats(task_model)
            loss_fn(task_model(inputs), labels).backward()
            optimiser.second_step(zero_grad=True)

        # Track the metrics for the batch (no gradients tracking)
        with torch.no_grad():
            _losses.append(loss.item())
            for metric_name, metric_fn in metrics.items():
                _metrics[metric_name].append(metric_fn(outputs, labels).item())

    # Clean up unused gpu memory
    optimiser.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    return np.array(_losses), {m: np.array(_metrics[m]) for m in _metrics}


def train_epoch_with_distillation(
    base_model: nn.Module,
    task_model: nn.Module,
    data_loader: DataLoader,
    optimiser: optim.Optimizer,
    loss_fn: Callable,
    metrics: Dict[str, Callable],
    device: torch.device,
    temperature: float,
    alpha: float,
    show_progress: bool = True,
) -> nn.Module:
    """
    Performs knowledge distillation from a base model to a task model.

    ### Args:
        `base_model (nn.Module)`: Teacher model to distill knowledge from
        `task_model (nn.Module)`: Student model to transfer knowledge to
        `task_loader (DataLoader)`: DataLoader containing the task data
        `device (torch.device)`: Device to perform computations on
        `temperature (float)`: Temperature parameter for softening probability distributions
        `num_epochs (int)`: Number of epochs to perform distillation

    ### Returns:
        `nn.Module`: The trained student model

    ### Note:
        The function uses only the soft targets for distillation:
        - Soft targets: KL divergence between teacher and student softened predictions
    """

    # Prepare models for distillation
    base_model.eval()
    task_model.train()

    # Move the base model to device ( task is already here )
    base_model = base_model.to(device)

    # Loss functions
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    def _distillation_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the distillation loss combining soft and hard targets.

        ### Args:
            `student_logits`: Raw output from student model
            `teacher_logits`: Raw output from teacher model
            `labels`: True labels
            `temperature`: Softmax temperature
            `alpha`: Weight for soft targets

        ### Returns:
            `torch.Tensor`: Combined distillation loss
        """
        soft_targets = (teacher_logits / temperature).softmax(dim=1)
        soft_prob = (student_logits / temperature).log_softmax(dim=1)

        # Both distillation losses
        soft_loss = kl_loss(soft_prob, soft_targets) * (temperature**2)
        hard_loss = loss_fn(student_logits, labels)
        return alpha * soft_loss + (1 - alpha) * hard_loss

    # Keep track of the losses and metrics
    _losses, _metrics = [], {metric: [] for metric in metrics}

    # Training loop
    data_iterator = tqdm(data_loader, desc="Training") if show_progress else data_loader
    for inputs, labels in data_iterator:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # When using SAM, we enable back the running stats.
        if not isinstance(optimiser, SAM):
            optimiser.zero_grad(set_to_none=True)
        else:
            enable_running_stats(task_model)

        # Compute combined loss
        with torch.no_grad():
            teacher_logits = base_model(inputs)

        student_logits = task_model(inputs)
        loss = _distillation_loss(
            student_logits,
            teacher_logits,
            labels,
        )
        loss.backward()

        # Optimisation step, in SAM we use the double step.
        if not isinstance(optimiser, SAM):
            optimiser.step()
        else:
            optimiser.first_step(zero_grad=True)
            disable_running_stats(task_model)
            _distillation_loss(
                task_model(inputs),
                teacher_logits,
                labels,
            ).backward()
            optimiser.second_step(zero_grad=True)

        # Track the metrics for the batch (no gradients tracking)
        with torch.no_grad():
            _losses.append(loss_fn(student_logits, labels).item())
            for metric_name, metric_fn in metrics.items():
                _metrics[metric_name].append(metric_fn(student_logits, labels).item())

    # Move base model back to cpu
    base_model = base_model.cpu()

    # Clean up unused gpu memory
    optimiser.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    return np.array(_losses), {m: np.array(_metrics[m]) for m in _metrics}


def validation_loss(
    model: nn.Module,
    task_loader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
) -> float:
    """
    Computes the average loss value of the model over the specific task.

    ### Args:
        `model (nn.Module)`: The model to validate.
        `task_loader (DataLoader)`: DataLoader for validation data.
        `loss_fn (Callable)`: Loss function to use for validation.

    ### Returns:
        `float`: Average loss over the validation set.
    """
    model.eval()

    # Iterate over the dataset without grmetrics tracking
    batch_losses = []
    with torch.no_grad():
        for inputs, labels in task_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_losses.append(loss_fn(outputs, labels).item())

        # Clean up unused gpu memory
        torch.cuda.empty_cache()

    return batch_losses


def validation_metrics(
    model: nn.Module,
    task_loader: DataLoader,
    metrics: Dict[str, Callable],
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Computes the validation metrics over the specific task.

    ### Args:
        `model (nn.Module)`: The model to validate.
        `task_loader (DataLoader)`: DataLoader for validation data.
        `metrics (Dict[str, Callable])`: Dictionary of metric functions.

    ### Returns:
        `Dict[str, float]`: Dictionary of average custom metrics over the validation set.
    """
    model.eval()
    _metrics = {metric: [] for metric in metrics}

    # Iterate over the dataset without gradients tracking
    with torch.no_grad():
        for inputs, labels in task_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Compute metrics
            for metric_name, metric_fn in metrics.items():
                _metrics[metric_name].append(metric_fn(outputs, labels).item())

        # Clean up unused gpu memory
        torch.cuda.empty_cache()

    return {m: np.array(_metrics[m]) for m in _metrics}


def get_optimiser(
    optim_name: str,
    optim_params: Dict[str, Any],
    model_parameters: Iterable[torch.nn.Parameter],
) -> optim.Optimizer:
    """
    Initialize and return the specified optimiser.

    ### Args:
        `optim_name (str)`: Name of the optimiser to initialize. Supported optimisers include:
            - 'sgd'
            - 'adam'
            - 'adamw'
            - 'rmsprop'
        `optim_params (Dict[str, Any])`: Parameters specific to the optimiser.
        `model_parameters (Iterable[torch.nn.Parameter])`: Parameters of the model to optimize.

    ### Returns:
        `optim.Optimizer`: An instance of the specified optimiser.

    ### Raises:
        `ValueError`: If the specified optimiser name is not supported.
    """
    optimisers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "rmsprop": optim.RMSprop,
        "sam": SAM,
    }

    optim_name_lower = optim_name.lower()
    if optim_name_lower not in optimisers:
        raise ValueError(f"Unsupported optimiser name '{optim_name}'")

    # Add some additional parameters depending on the optimiser
    if optim_name_lower == "sam":
        optim_params["base_optimizer"] = optim.SGD

    OptimClass = optimisers[optim_name_lower]
    return OptimClass(model_parameters, **optim_params)
