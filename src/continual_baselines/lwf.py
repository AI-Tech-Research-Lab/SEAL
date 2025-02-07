import copy
from continual_learning.continual_problems import DataContinualTrainer

from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import torch

from typing import Dict, Callable, Tuple


class LwFBaselineTrainer(DataContinualTrainer):
    """
    Represents the Learning without Forgetting (LwF) baseline.
    Implements knowledge distillation to mitigate catastrophic forgetting.
    """

    def __init__(
        self,
        *args,
        lwf_lambda: float = 1.0,
        temperature: float = 2.0,
        alpha: float = 0.75,
        **kwargs,
    ):
        """
        Initializes the LwFBaselineTrainer.

        Args:
            lwf_lambda (float, optional): Regularization strength for LwF. Defaults to 1.0.
            temperature (float, optional): Temperature parameter for softening probabilities. Defaults to 2.0.
            alpha (float, optional): Weighting factor between classification loss and distillation loss. Defaults to 0.5.
        """
        super().__init__(*args, **kwargs)
        self.lwf_lambda = lwf_lambda
        self.temperature = temperature
        self.alpha = alpha

        # Store a copy of the previous model for generating soft targets
        self.old_model: nn.Module = None
        self.old_model_requires_grad: bool = False

        # Base loss functions
        self.base_loss_fn = nn.CrossEntropyLoss()
        self.distillation_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def get_task_model(
        self,
        task_id: str,
        base_model: nn.Module,
        training: bool = True,
    ) -> nn.Module:
        """
        All the tasks are evaluated with the same model.
        NOTE: the base model is RE-TRAINED with each new task it sees.

        Args:
            task_id (str): Identifier for the current task.
            base_model (nn.Module): The neural network model.
            training (bool, optional): Flag indicating if the model is in training mode. Defaults to True.

        Returns:
            nn.Module: The model to be trained on the current task.
        """
        return base_model

    def store_old_model(self, model: nn.Module):
        """
        Stores a copy of the current model to be used for generating soft targets.

        Args:
            model (nn.Module): The neural network model.
        """

        # Store a copy of the current model
        self.old_model = copy.deepcopy(model)

        # Set the model to not require gradients
        for param in self.old_model.parameters():
            param.requires_grad = False

        # Move the model to the correct device and set to eval
        self.old_model.to(self.device)
        self.old_model.eval()

    def lwf_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the combined loss with LwF regularization.

        Args:
            model (nn.Module): The neural network model.
            outputs (torch.Tensor): The model outputs for the current task.
            targets (torch.Tensor): The target labels.
            inputs (torch.Tensor): The input data batch.

        Returns:
            torch.Tensor: The combined loss with LwF regularization.
        """
        # Classification loss
        classification_loss = self.base_loss_fn(outputs, targets)

        # Distillation loss
        if self.old_model is not None:
            # Compute logits based on old model
            with torch.no_grad():
                old_outputs = self.old_model(inputs)

            # Soft targets are the comparion of the logits of the models
            soft_targets = nn.functional.softmax(old_outputs / self.temperature, dim=1)
            new_outputs = nn.functional.log_softmax(outputs / self.temperature, dim=1)

            # Compute distillation loss
            distillation_loss = self.distillation_loss_fn(new_outputs, soft_targets)
            distillation_loss *= self.temperature**2
            distillation_loss *= self.lwf_lambda
        else:
            distillation_loss = 0.0

        # Combined loss
        combined_loss = (
            self.alpha * classification_loss + (1 - self.alpha) * distillation_loss
        )
        return combined_loss

    def get_training_routine(self) -> Callable:
        """
        Returns the training routine function that incorporates LwF loss.

        Returns:
            Callable: Training function for each epoch.
        """

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
            Performs a single training epoch using LwF loss.

            Args:
                task_model (nn.Module): The neural network model.
                data_loader (DataLoader): DataLoader for the training data.
                optimiser (optim.Optimizer): The optimizer.
                loss_fn (Callable): The primary loss function.
                metrics (Dict[str, Callable]): Dictionary of metric functions.
                device (torch.device): The device to perform computations on.
                show_progress (bool, optional): Whether to display a progress bar. Defaults to True.

            Returns:
                Tuple[np.ndarray, Dict[str, np.ndarray]]: Training losses and metrics.
            """
            task_model.train()
            _losses, _metrics = [], {metric: [] for metric in metrics}

            data_iterator = (
                tqdm(data_loader, desc="Training") if show_progress else data_loader
            )

            for inputs, labels in data_iterator:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimiser.zero_grad()

                outputs = task_model(inputs)
                loss = self.lwf_loss(outputs, labels, inputs)
                loss.backward()
                optimiser.step()

                with torch.no_grad():
                    _losses.append(loss.item())
                    for metric_name, metric_fn in metrics.items():
                        _metrics[metric_name].append(metric_fn(outputs, labels).item())

            # Clean up unused GPU memory
            optimiser.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            return np.array(_losses), {m: np.array(_metrics[m]) for m in _metrics}

        return train_epoch

    def model_task_change(self, task_id: str, task_model: nn.Module):
        """
        Handle task changes by updating the old model and storing parameters.

        Args:
            task_id (str): Identifier for the current task.
            task_model (nn.Module): The trained model for the current task.
        """
        # Store the current model as the old model for future distillation
        self.store_old_model(task_model)

        # Update base model
        self.base_model = task_model

    def train(self, *args, **kwargs):
        """
        Initializes LwF structures before training.
        """
        # Initialize previous task parameters as None
        self.old_model = None
        return super().train(*args, **kwargs)
