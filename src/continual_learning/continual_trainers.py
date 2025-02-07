from functools import partial
from continual_learning.continual_problems import DataContinualTrainer
from search_space._ofa_expansions import apply_model_expansion
from search_space import get_ofa_evaluator, get_search_space

from continual_learning._training import (
    train_epoch_with_distillation,
    validation_loss,
    train_epoch,
)

from torch.utils.data import DataLoader
from torch import nn

import numpy as np

from typing import Callable


class FixedDataContinualTrainer(DataContinualTrainer):
    """
    Represent the base problem where the model is never expanded.
    Used as a baseline to compare the effectiveness of the expansions within the NAS.
    """

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


class GrowingDataContinualTrainer(DataContinualTrainer):
    """
    Represents the main problem for the thesis. The model is expanded, depending on its
    capacity for the new task, every time a task arrives.
    """

    def __init__(
        self,
        capacity_tau: float,
        expand_is_frozen: bool = False,
        distill_on_expand: bool = True,
        weights_from_ofa: bool = True,
        dataset_name: str = "cifar10",
        search_space_family: str = "mobilenetv3",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Define the capacity threshold
        self.capacity_tau = capacity_tau

        # Expanded model freezes previous weights or allows them to be updated
        self.expand_is_frozen = expand_is_frozen

        # Perform knowledge distillation for the model expansion
        self.distill_on_expand = distill_on_expand
        self.model_was_expanded = False

        # Parameters for the distiallation loss
        self.logits_temperature = 2.0
        self.distillation_weight = 0.2

        # Define the space on which to expand the model
        self.search_space = get_search_space(search_space_family)
        self.ofa_evaluator = get_ofa_evaluator(
            search_space_family,
            dataset_name,
            pretrained=weights_from_ofa,
        )

    def get_training_routine(self) -> Callable:
        """
        Add the distillation training when the model has been recently expanded.
        """
        if self.distill_on_expand and self.model_was_expanded:
            training_routine = partial(
                train_epoch_with_distillation,
                base_model=self.base_model,
                temperature=self.logits_temperature,
                alpha=self.distillation_weight,
            )
            return training_routine
        return train_epoch

    def get_model_capacity(self, model: nn.Module, task_loader: DataLoader) -> int:
        """
        Computes the model's capacity on the incoming task.

        The current implementation for capacity computes the loss value over the new task.
        > As capacity is a proxy, we are using only one iteration over the task's data.
        """

        # Move the model to the device for computation
        model = model.to(self.device)

        # Compute the incoming task loss
        incoming_task_losses = validation_loss(
            model=model,
            task_loader=task_loader,
            loss_fn=self.loss_fn,
            device=self.device,
        )

        # Move model back to CPU
        model = model.cpu()

        # Get the training losses from the last iteration.
        running_losses = np.array(self.last_training_losses)

        # The capacity is computed from the difference in the losses.
        capacity_metric = np.median(running_losses) - np.median(incoming_task_losses)
        capacity_metric = capacity_metric / np.median(running_losses)
        return capacity_metric

    def get_task_model(
        self,
        task_id: str,
        base_model: nn.Module,
        training: bool = True,
    ) -> nn.Module:
        """
        Performs the model expansion when needed by the model's capacity on the incoming tasks.

        The steps to check and perform the expansion are the following:
        - Computes the current model's capacity on the incoming task.
        - If the capacity is under the defined threshold, expand the model.
        - Model expansion tries to preserve as many as the original model weights.
        - Expanded model is fine-tuned distilling from base model.
        """
        self.model_was_expanded = False

        # When in evaluation mode, we just return the current base model
        if not training:
            return base_model

        # When where are encountering the first task, we just return the base model
        if (task_id == 0) or (task_id == "0"):
            return base_model

        # Get task incoming training data
        task_loader = self.dataset.get_task_loader(
            task_id=task_id,
            resolution=self.current_resolution,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            augment=False,
            shuffle=True,
        )

        # Check if the models needs to be expan
        base_capacity = self.get_model_capacity(
            model=base_model,
            task_loader=task_loader,
        )

        if base_capacity <= self.capacity_tau:
            # Expand the model with the scaling direction
            task_model_definition, task_model = apply_model_expansion(
                base_model=base_model,
                model_sample=self.model_definition,
                search_space=self.search_space,
                ofa_evaluator=self.ofa_evaluator,
                scaling_direction=self.model_definition["direction"],
                freeze_base=self.expand_is_frozen,
            )

            # Check that the expanded architecture produced new parameters
            _trainable_params = sum(
                p.numel() for p in task_model.parameters() if p.requires_grad
            )

            # The following code is needed as there are some OFA archicture definitions
            # that collide between each other ( different definitions for the same architecture).
            # To avoid expanding into this architectures, when no trainable parameters are added
            # to the model, the architecture is not updated.
            if _trainable_params == 0:
                print("No trainable parameters, returning base model")
                print(f"Initial model definition: {self.model_definition}")
                print(f"Model definition: {task_model_definition}")
                task_model = base_model

            # Store the updated model definition
            self.model_definition = task_model_definition

            # Set the expanded flag to utilse the correct loss function
            self.model_was_expanded = True

            return task_model

        return base_model
