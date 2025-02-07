from continual_learning._training import validation_metrics, get_optimiser, train_epoch
from continual_learning._flatness import calculate_mean_flatness
from continual_learning.logging import ContinualTrainerLogger
from continual_learning.base_dataset import ContinualDataset
from search_space import ModelSample

from tools.data_utils import get_dataset_from_name
from _utils import set_seed


from abc import ABC, abstractmethod
from copy import deepcopy
import torch.nn as nn
import torch
import gc

from typing import Callable, Dict, List, ClassVar


class ContinualModelTrainer(ABC):
    data_class = ClassVar[ContinualDataset]

    def __init__(
        self,
        experiment_dir: str,
        experiment_name: str,
        model_definition: ModelSample,
        base_model: nn.Module,
        num_tasks: int,
        random_seed: int,
    ):
        """
        Initialize the ContinualModelTrainer.

        ### Args:
            `base_dataset (Dataset)`: The base dataset.
            `model_definition (Sample)`: The model definition from the search space.
            `base_model (nn.Module)`: The base model to train.
            `num_tasks (int)`: Number of tasks in continual learning.
            `random_seed (int)`: Seed for reproducibility.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = base_model

        # This values are associated with the base model
        self.model_definition = model_definition
        self.current_resolution = self.model_definition["resolution"]

        # Data transformation for the tasks
        self.num_tasks = num_tasks

        # Manage reproducibility of data
        self.random_seed = random_seed
        set_seed(self.random_seed)

        # Initialize metrics and logging structure for tracking
        self.logger = ContinualTrainerLogger(
            experiment_dir=experiment_dir,
            experiment_name=experiment_name,
        )

        # The `prepare_tasks` method should set these parameters
        self.dataset_name: str
        self.dataset: ContinualDataset

        # Loading of the datasets
        self.batch_size: int
        self.num_workers: int

        # These are set by the `set_experiment_settings`:
        self.augment: bool
        self.loss_fn: Callable
        self.training_metrics: Dict[str, Callable]

        # Some methods depend on training statistics
        self.last_training_losses: List[float] = []

        # Flag to compute the flatness of the model's weights in some test cases
        self.always_compute_flatness: bool = False

    @abstractmethod
    def get_task_model(
        self,
        task_id: str,
        base_model: nn.Module,
        training: bool = True,
    ) -> nn.Module:
        """
        Returns the model to use on a specific task.

        This produces a model with the correct `model-head` for the specific task.
        """
        pass

    @abstractmethod
    def model_task_change(self, task_id: str, task_model: nn.Module):
        """
        Updates the base_model based on the `task_model`. This method implementation will
        change depending on the type of continual learning we are using.
        """
        pass

    def on_task_complete(self, task_id: str):
        """
        When a task is completed we need to update the base model accordingly. This function
        contains the necessary operations to do keep the task-agnostic ( something like the backbone )
        of the model.

        > Logging the change and tracking data changes
        """
        self.logger.complete_task(task_id)

        # Force memory collection
        torch.cuda.empty_cache()
        gc.collect()

    def on_training_complete(self) -> str:
        """
        Handles operations to perform when the training is complete.
        """
        self.logger.complete_task("training_metrics")
        self.logger.store_run()

        # Clean up the memory
        torch.cuda.empty_cache()
        gc.collect()

    def get_training_routine(self) -> Callable:
        """
        This is intended to be overwritten if needed.
        """
        return train_epoch

    def _validate_task(
        self,
        task_id: str,
        task_name: str,
        task_model: nn.Module,
        batch_size: int,
        num_workers: int = 1,
        end_of_training=False,
    ) -> None:
        """
        Validate the model on a specific task and log the metrics.

        ### Args:
            `task_id (str)`: Identifier of the task to validate.
            `task_model (nn.Module)`: The model to validate.
            `batch_size (int)`: Batch size for the validation data loader.
            `num_workers (int)`: Number of workers for the validation data loader.
        """

        # Get task-specific validation data loader
        val_loader = self.dataset.get_task_loader(
            task_id=task_id,
            resolution=self.current_resolution,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=False,
            shuffle=False,
        )

        # Compute the loss and metrics for the task
        task_metrics = validation_metrics(
            model=task_model.to(self.device),
            task_loader=val_loader,
            metrics=self.training_metrics,
            device=self.device,
        )

        # Compute the flatness of the model's weights only when
        # we are not evaluating the full model ( only for NAS evaluation )
        if end_of_training or self.always_compute_flatness:
            model_flatness = calculate_mean_flatness(
                task_model,
                val_loader,
                sigma=0.05,
                device=self.device,
                neighbourhood_size=5,
            )
            task_metrics["flatness"] = model_flatness

        # Update the logger with validation metrics
        self.logger.log_validation_metrics(
            task_id=task_name,
            metrics=task_metrics,
        )

        # Move the model back to CPU
        task_model = task_model.to("cpu")

        return task_model

    def _train_task(
        self,
        task_id: str,
        task_model: nn.Module,
        n_epochs: int,
        batch_size: int,
        num_workers: int = 1,
        show_progress: bool = True,
    ):
        """
        Train the model on a specific task.
        """

        # Move the model to the GPU
        train_task_model = task_model.to(self.device)

        # Get task specifc training data loader
        train_loader = self.dataset.get_task_loader(
            task_id=task_id,
            resolution=self.current_resolution,
            batch_size=batch_size,
            augment=self.augment,
            num_workers=num_workers,
            shuffle=True,
        )

        # Initialise an optimiser for the model.
        optimiser = get_optimiser(
            optim_name=self.optimiser_name,
            optim_params=self.optimiser_params,
            model_parameters=train_task_model.parameters(),
        )

        # Get the correct training function to use depending on the type of training
        training_routine = self.get_training_routine()

        for i in range(n_epochs):
            train_losses, train_metrics = training_routine(
                task_model=train_task_model,
                data_loader=train_loader,
                optimiser=optimiser,
                loss_fn=self.loss_fn,
                metrics=self.training_metrics,
                device=self.device,
                show_progress=show_progress,
            )

            # Update the logger training history
            self.logger.log_training_metrics(
                train_losses,
                train_metrics,
                print_metrics=show_progress,
            )

            # Track the initial running training loss
            if i == 0:
                self.last_training_losses = train_losses

        # Move the model to the CPU and return it
        train_task_model = train_task_model.cpu()
        torch.cuda.empty_cache()

        return train_task_model

    def validate(self, end_of_training: bool = False):
        """
        Validate the model across all the dataset tasks.
        """
        for tid in self.dataset.tasks.keys():
            task_model = self.get_task_model(
                tid,
                self.base_model,
                training=False,
            )

            self._validate_task(
                task_id=tid,
                task_name=f"cont_{tid + 1}",
                task_model=task_model,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                end_of_training=end_of_training,
            )

    def random_train(
        self,
        task_epochs: List[int] | int,
        show_progress: bool = True,
    ):
        """
        This is use as a baseline of each of the tasks. To compare the continual learning and
        the effects on learning the tasks, we evaluate a "random" initilisation for each
        of the tasks.

        ### Args:
            `batch_size (int)`: Batch size for the training data loaders.
            `task_epochs (List[int] | int)`: Number of epochs to train on each task. If a single
            int is provided, the same number of epochs is used for all tasks.
            `num_workers (int)`: Number of workers for the training data loaders.
        """
        if isinstance(task_epochs, int):
            epochs_per_task = [task_epochs] * self.num_tasks
        else:
            epochs_per_task = task_epochs

        for tid, epochs in zip(self.dataset.tasks.keys(), epochs_per_task):
            task_name = f"random_{tid + 1}"

            # Get the independant instance of the model
            self.base_model = self.base_model.cpu()
            _base_model = deepcopy(self.base_model)

            # Create the task model from the independant instance
            random_task_model = self.get_task_model(
                task_id=tid,
                base_model=_base_model,
                training=False,
            )

            # Get model specifc training data loader
            random_task_model = self._train_task(
                task_id=tid,
                task_model=random_task_model,
                n_epochs=epochs,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                show_progress=show_progress,
            )

            # Validate the model
            self._validate_task(
                task_id=tid,
                task_name=task_name,
                task_model=random_task_model,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                end_of_training=False,
            )

            # Performs the logging and clean up of stored values
            self.on_task_complete(task_name)

    def continual_train(
        self,
        task_epochs: List[int] | int,
        show_progress: bool = True,
        evaluate_after_task: bool = False,
    ):
        """
        Implements the whole continual model training pipeline. This process follows
        the task incremental paradigm.

        We train on each task and, at the end of each epoch, we evaluate the model on
        all the available tasks.

        ### Args:
            `batch_size (int)`: Batch size for the training data loaders.
            `task_epochs (List[int] | int)`: Number of epochs to train on each task. If a single
            int is provided, the same number of epochs is used for all tasks.
            `num_workers (int)`: Number of workers for the training data loaders.
            `evaluate_after_task (bool)`: Whether to evaluate the model after each task.
        """
        if isinstance(task_epochs, int):
            epochs_per_task = [task_epochs] * self.num_tasks
        else:
            epochs_per_task = task_epochs

        for tid, epochs in zip(self.dataset.tasks.keys(), epochs_per_task):
            task_name = f"cont_{tid + 1}"

            # Get the specific model for the current task
            continual_task_model = self.get_task_model(
                task_id=tid,
                base_model=self.base_model,
            )

            # Get model specifc training data loader
            continual_task_model = self._train_task(
                task_id=tid,
                task_model=continual_task_model,
                n_epochs=epochs,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                show_progress=show_progress,
            )

            # Updates the self.base_model with the trained task model
            self.model_task_change(
                task_id=tid,
                task_model=continual_task_model,
            )

            # When finishing training on the task, we compute the "test" metrics
            # on all the tasks.
            if evaluate_after_task:
                self.validate()

            # Performs the logging and clean up of stored values
            self.on_task_complete(task_name)

    def load_dataset(
        self,
        dataset_name: str,
        evaluation: bool = False,
        normalise: bool = True,
    ):
        """
        Prepares the continual dataset and their according continual models heads.

        Note: this method should set the parameter `self.dataset` with a
        `ContinualDataset` class.
        """
        base_dataset, mean, std = get_dataset_from_name(
            dataset_name,
            evaluation=evaluation,
        )

        # Load the continual dataset
        self.dataset_name = dataset_name
        self.dataset = self.data_class(
            dataset=base_dataset,
            num_tasks=self.num_tasks,
            normalise=normalise,
        )

        # Store the data parameters
        self.dataset.set_data_params(mean=mean, std=std)

    def set_experiment_settings(
        self,
        loss_fn: Callable,
        epochs_per_task: List[int] | int,
        optim_name: str,
        optim_params: Dict[str, float],
        model_size_metrics: Dict[str, float],
        training_metrics: Dict[str, Callable],
        augment: bool,
    ):
        """
        Setups some training parameters that are unchanged during the evaluation
        (for both the random and the training).

        ### Args:
            `optim_name (str)`: Name of the optimiser to use.
            `optim_params (Dict[str, float])`: Parameters for the optimiser.
            `loss_fn (Callable)`: Loss function to use.
            `training_metrics (Dict[str, Callable])`: Metrics to track during training.
            `augment (bool)`: Whether to use data augmentation.
        """
        self.augment = augment
        self.loss_fn = loss_fn
        self.optimiser_name = optim_name
        self.optimiser_params = optim_params
        self.training_metrics = training_metrics
        self.logger.store_settings(
            settings={
                "augment": augment,
                "n_tasks": self.num_tasks,
                "optimiser_name": optim_name,
                "optimiser_params": optim_params,
                "sample_definition": self.model_definition,
                "epochs_per_task": epochs_per_task,
                **model_size_metrics,
            }
        )

    def train(
        self,
        batch_size: int,
        task_epochs: List[int] | int,
        num_workers: int = 1,
        show_progress: bool = True,
        with_random_metrics: bool = False,
        evaluate_after_task: bool = False,
    ):
        """
        Main entrypoint to evaluate the model.
        """

        # Set the batch size and number of workers
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Perform random evaluation of the models
        if with_random_metrics:
            self.random_train(
                task_epochs=task_epochs,
                show_progress=show_progress,
            )

        # Perform the training in a continual learning manner
        self.continual_train(
            task_epochs=task_epochs,
            show_progress=show_progress,
            evaluate_after_task=evaluate_after_task,
        )

        # Compute the final task metrics
        self.validate(end_of_training=True)

        # Store the metrics and clean up
        self.on_training_complete()
