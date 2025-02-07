from continual_learning.base_dataset import DataSubset
from continual_learning.continual_problems import DataContinualTrainer
from continual_learning._training import get_optimiser
from tools.data_utils import get_transformation

from torch.utils.data import DataLoader, ConcatDataset
from torch import nn


class JointBaselineTrainer(DataContinualTrainer):
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

    def get_joint_datasets(self, batch_size: int, num_workers: int) -> DataLoader:
        """
        Get the training data for all the tasks.
        """
        # Builds the training transformation step from dataset params
        training_transformer = get_transformation(
            mean=self.dataset.DATA_MEAN,
            std=self.dataset.DATA_STD,
            resolution=self.current_resolution,
            augment=self.augment,
        )

        # Joint all the training data into a single DataLoader
        combined_dataset = ConcatDataset(
            [self.dataset.tasks[task_id] for task_id in self.dataset.tasks.keys()]
        )

        # Applit the transformation
        transformed_training_subset = DataSubset(
            combined_dataset,
            transform=training_transformer,
        )

        # Unique dataloader from the combined datasets
        train_loader = DataLoader(
            transformed_training_subset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )

        return train_loader

    def train(
        self,
        task_epochs: int,
        batch_size: int,
        num_workers: int = 1,
        show_progress: bool = True,
        # For compatibility with the other trainers
        with_random_metrics: bool = True,
        evaluate_after_task: bool = True,
    ):
        """Perform the training over all the training tasks at once."""

        # Start from the base model
        joint_tasks_model = self.base_model.to(self.device)

        # Get the joint training dataloader
        self.batch_size = batch_size
        self.num_workers = num_workers
        train_loader = self.get_joint_datasets(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        # Initialise an optimiser for the model.
        optimiser = get_optimiser(
            optim_name=self.optimiser_name,
            optim_params=self.optimiser_params,
            model_parameters=joint_tasks_model.parameters(),
        )

        # Define custom task_id
        task_id = "joint_training"

        # Get the correct training function to use depending on the type of training
        training_routine = self.get_training_routine()

        for _ in range(task_epochs):
            train_losses, train_metrics = training_routine(
                task_model=joint_tasks_model,
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

        # Updates the self.base_model with the trained task model
        self.model_task_change(
            task_id=task_id,
            task_model=joint_tasks_model,
        )

        # When finishing training on the task, we compute the "test" metrics
        # on all the tasks.
        self.validate(end_of_training=True)

        # Performs the logging and clean up of stored values
        self.on_task_complete(task_id)
        self.logger.store_run()

        return joint_tasks_model
