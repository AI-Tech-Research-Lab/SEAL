from continual_learning.continual_problems import DataContinualTrainer

from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import torch

from typing import Dict, Callable, Tuple


class SIBaselineTrainer(DataContinualTrainer):
    """
    Represent the SI baseline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_lambda = 1
        self.damping_factor = 0.1

        # Storing structures for SI
        self.omega: Dict[str, torch.Tensor] = {}  # Parameter importance
        self.prev_params: Dict[str, torch.Tensor] = {}  # Previous task parameters
        self.running_sum: Dict[str, torch.Tensor] = {}  # gradients * updates

        # Base loss function
        self.base_loss_fn = nn.CrossEntropyLoss()

    def get_task_model(
        self,
        task_id: str,
        base_model: nn.Module,
        training: bool = True,
    ) -> nn.Module:
        """
        All the tasks are evaluated with the same model.
        NOTE: the base model is RE-TRAINED with each new task it sees.
        """
        return base_model

    def initialise_task_tracking(self, model: nn.Module):
        """
        Re-initialises the tracking structures for the current task.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.running_sum[name] = torch.zeros_like(param, device=self.device)

    def update_importance_estimates(self, model: nn.Module):
        """
        Update running sum with current gradient and parameter update.

        Args:
            model (nn.Module): The neural network model
        """
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                delta_theta = param.detach() - self.prev_params[name]
                self.running_sum[name] += -param.grad.detach() * delta_theta

    def normalise_omega_l2(self):
        """
        Normalise the omega values by their L2 norm.
        """
        # Compute the total squared omega across all parameters
        total_squared = torch.tensor(0.0, device=self.device)
        for omega in self.omega.values():
            total_squared += torch.sum(omega.pow(2))

        # Add a small constant to avoid division by zero
        total = torch.sqrt(total_squared) + 1e-8

        # Normalize each omega tensor
        for name in self.omega:
            self.omega[name] = self.omega[name] / total

    def update_omega(self, model: nn.Module):
        """
        Computes the synaptic importance (omega) for the given model params.

        Args:
            model (nn.Module): The neural network model.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping parameter names to their omega estimates.
        """

        # Move model to device
        model = model.to(self.device)

        for name, param in model.named_parameters():
            if param.requires_grad:
                delta_theta = param.detach() - self.prev_params[name]
                denominator = delta_theta.pow(2) + self.damping_factor

                # Initialisation of omega
                if name not in self.omega:
                    self.omega[name] = torch.zeros_like(param, device=self.device)

                # Update omega using path integral
                self.omega[name].add_(self.running_sum[name] / denominator)

        # Move the model back to the CPU
        model = model.to("cpu")

        # Normalise the omega values by their L2 norm
        # This is not in the original paper
        self.normalise_omega_l2()

    def si_loss(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the total loss with SI regularization.

        Args:
            model (nn.Module): The neural network model
            outputs (torch.Tensor): Model outputs
            targets (torch.Tensor): Target labels

        Returns:
            torch.Tensor: Combined loss with SI regularization
        """
        # Task-specific loss
        base_loss = self.base_loss_fn(outputs, targets)

        # SI regularization
        si_reg = 0
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.prev_params:
                si_reg += (
                    self.omega[name] * (param - self.prev_params[name]).pow(2)
                ).sum()

        return base_loss + (self.si_lambda / 2) * si_reg

    def get_training_routine(self) -> Callable:
        """
        Returns the training routine function that incorporates SI loss and updates.
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
                loss = self.si_loss(task_model, outputs, labels)
                loss.backward()

                # Update running sum before optimizer step - we need the grad
                self.update_importance_estimates(task_model)
                optimiser.step()

                with torch.no_grad():
                    _losses.append(loss.item())
                    for metric_name, metric_fn in metrics.items():
                        _metrics[metric_name].append(metric_fn(outputs, labels).item())

            optimiser.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            return np.array(_losses), {m: np.array(_metrics[m]) for m in _metrics}

        return train_epoch

    def model_task_change(self, task_id: str, task_model: nn.Module):
        """
        Handle task changes by updating omega and storing parameters.
        """
        # Compute final omega values for the completed task
        self.update_omega(task_model)

        # Store current parameters for next task's regularization
        self.prev_params = {
            name: param.clone().detach().to(self.device)
            for name, param in task_model.named_parameters()
            if param.requires_grad
        }

        # Update base model
        self.base_model = task_model

    def train(self, *args, **kwargs):
        """
        Initialize SI structures before training.
        """
        # Initialize omega if first task
        self.omega = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.base_model.named_parameters()
            if param.requires_grad
        }

        # Initialise the previous task parameters as zeros
        self.prev_params = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.base_model.named_parameters()
            if param.requires_grad
        }

        # Initialize tracking for the new task
        self.initialise_task_tracking(self.base_model)
        return super().train(*args, **kwargs)
