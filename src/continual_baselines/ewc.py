from continual_learning.continual_problems import DataContinualTrainer

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import torch

from typing import Dict, Callable, Tuple


class EWCBaselineTrainer(DataContinualTrainer):
    """
    Represent the EWC baseline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = 1

        # Storing structures for the EWC-Loss
        self.fisher: Dict[str, torch.Tensor] = {}
        self.previous_task_params: Dict[str, torch.Tensor] = {}

        # Define a base loss function
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

    def compute_fisher(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        sample_size: int = 1024,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the diagonal Fisher Information Matrix for the given model.

        Args:
            model (nn.Module): The neural network model.
            data_loader (DataLoader): DataLoader for the training data of the current task.
            device (torch.device): The device to perform computations on.
            sample_size (int, optional): Number of samples to use for estimating FIM. Defaults to 1024.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping parameter names to their FIM estimates.
        """
        model.eval()

        # Move model to device
        model = model.to(self.device)

        # Placeholder for the Fisher Information Matrix
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Iterate over the data loader
        data_iterator = iter(data_loader)

        # Cap the number of samples to use for estimating the FIM
        num_samples = 0
        while num_samples < sample_size:
            try:
                inputs, targets = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                inputs, targets = next(data_iterator)

            # Compute the negative log-likelihood
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            model.zero_grad()
            outputs = model(inputs)
            log_probs = F.log_softmax(outputs, dim=1)
            loss = F.nll_loss(log_probs, targets, reduction="sum")
            loss.backward()

            # Compute the Fisher Information Matrix
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().clone().pow(2)

            num_samples += inputs.size(0)

        # Normalize FIM
        for name in fisher:
            fisher[name] = fisher[name] / num_samples

        return fisher

    def store_previous_task_params(self, model: nn.Module):
        """
        Stores a copy of the model's current parameters.

        Args:
            model (nn.Module): The neural network model.
        """
        model.eval()
        optimal = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.previous_task_params = optimal

    def model_task_change(self, task_id: str, task_model: nn.Module):
        """
        Overwrites the base model and computes the Fisher Information Matrix using
        the same training data as the current task.
        """

        # Store the task model as the base model
        self.base_model = task_model

        # Get the current task training data
        train_loader = self.dataset.get_task_loader(
            task_id=task_id,
            resolution=self.current_resolution,
            batch_size=self.batch_size,
            augment=self.augment,
            num_workers=self.num_workers,
            shuffle=True,
        )

        # Compute the Fisher Information Matrix using the training data of the current task
        self.fisher = self.compute_fisher(
            model=task_model,
            data_loader=train_loader,
            sample_size=int(0.2 * len(train_loader.dataset)),
        )

        # Store the model optimal parameters
        self.store_previous_task_params(task_model)

    def ewc_loss(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the EWC regularization loss.

        Args:
            model (nn.Module): The neural network model.
            outputs (torch.Tensor): The model outputs.
            targets (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The combined loss with EWC regularization.
        """
        base_loss = self.base_loss_fn(outputs, targets)

        # EWC-Loss computation
        ewc_loss = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                ewc_loss += (
                    self.fisher[name] * (param - self.previous_task_params[name]).pow(2)
                ).sum()

        return base_loss + (self.ewc_lambda / 2) * ewc_loss

    def get_training_routine(self) -> Callable:
        """
        Returns the training routine function that incorporates EWC loss.

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
            Performs a single training epoch using EWC loss.

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

            # Keep track of the losses and metrics
            _losses, _metrics = [], {metric: [] for metric in metrics}

            # Iterate over the dataset
            data_iterator = (
                tqdm(data_loader, desc="Training") if show_progress else data_loader
            )
            for inputs, labels in data_iterator:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Clean accumulated gradients
                optimiser.zero_grad()

                # Forward pass
                outputs = task_model(inputs)
                loss = self.ewc_loss(task_model, outputs, labels)

                # Backward pass
                loss.backward()

                # Optimization step
                optimiser.step()

                # Track the metrics for the batch
                with torch.no_grad():
                    _losses.append(loss.item())
                    for metric_name, metric_fn in metrics.items():
                        _metrics[metric_name].append(metric_fn(outputs, labels).item())

            # Clean up unused GPU memory
            optimiser.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            return np.array(_losses), {m: np.array(_metrics[m]) for m in _metrics}

        return train_epoch

    def train(self, *args, **kwargs):
        """
        Prepares the structures for the EWC baseline.
        """

        # Initialise the fisher matrix with zeros
        self.fisher = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.base_model.named_parameters()
            if param.requires_grad
        }

        # Initialise the previous task parameters
        self.previous_task_params = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.base_model.named_parameters()
            if param.requires_grad
        }

        return super().train(*args, **kwargs)
