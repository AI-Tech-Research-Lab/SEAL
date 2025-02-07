# ruff: noqa: F401
import _config

from continual_learning.continual_trainers import (
    FixedDataContinualTrainer,
    GrowingDataContinualTrainer,
)
from search_space import get_search_space, get_ofa_evaluator
from search_space.ofa_tools import get_net_info
from tools.metrics import binary_accuracy
from _utils import set_seed

from functools import partial
import torch.nn as nn
import argparse
import os

OFA_FAMILY = "mobilenetv3"


TAUS = {
    "cifar10": 0.2,
    "cifar100": 0.13,
    "imagenet16": 0.13,
}


def get_continual_trainer(
    dataset_name: str,
    fixed_architecture: bool = False,
) -> FixedDataContinualTrainer | GrowingDataContinualTrainer:
    if fixed_architecture:
        return FixedDataContinualTrainer

    return partial(
        GrowingDataContinualTrainer,
        capacity_tau=TAUS[dataset_name],
        expand_is_frozen=False,
        distill_on_expand=True,
        weights_from_ofa=True,
        dataset_name=dataset_name,
        search_space_family=OFA_FAMILY,
    )


def main(
    model_encoding: str,
    experiment_dir: str,
    n_tasks: int,
    optimiser_name: str,
    learning_rate: float,
    weight_decay: float,
    epochs_per_task: int,
    random_seed: int,
    dataset_name: str,
    fixed_architecture: bool = False,
) -> None:
    """
    Main function to initialize the trainer and start the training process.
    """
    set_seed(random_seed)

    # Get the search space and the OFA supernet
    supernet = get_ofa_evaluator(family=OFA_FAMILY, dataset=dataset_name)
    search_space = get_search_space(
        family=OFA_FAMILY,
        fixed=fixed_architecture,
    )

    # Parse to int all the values to ensure the correct decoding
    sampled_architecture = search_space.decode(model_encoding)
    base_model = supernet.get_architecture_model(sampled_architecture)

    # Get the model size-related metrics
    model_resolution = sampled_architecture["resolution"]
    model_size_metrics = get_net_info(
        base_model,
        input_shape=(3, model_resolution, model_resolution),
    )

    # Get the model folder name
    model_name = "".join(map(str, model_encoding))
    experiment_name = f"{model_name}-{random_seed}"

    # Check if the model has already been trained
    if os.path.exists(
        f"{experiment_dir}-{n_tasks}/models/{experiment_name}/history.json"
    ):
        print(f"Model {experiment_name} already trained, skipping...")
        return

    # Load the trainer with the dataset
    continual_trainer = get_continual_trainer(
        dataset_name=dataset_name,
        fixed_architecture=fixed_architecture,
    )

    # Load the continual trainer
    trainer = continual_trainer(
        experiment_dir=f"{experiment_dir}-{n_tasks}",
        experiment_name=experiment_name,
        model_definition=sampled_architecture,
        base_model=base_model,
        num_tasks=n_tasks,
        random_seed=random_seed,
    )

    # Load the data and the stats to use for normalisation
    trainer.load_dataset(dataset_name=dataset_name, evaluation=True)

    # Set training settings
    trainer.set_experiment_settings(
        loss_fn=nn.CrossEntropyLoss(),
        epochs_per_task=epochs_per_task,
        optim_name=optimiser_name,
        optim_params={"lr": learning_rate, "weight_decay": weight_decay},
        model_size_metrics=model_size_metrics,
        training_metrics={"accuracy": binary_accuracy},
        augment=False,
    )

    # Train the model (random and continual)
    trainer.train(
        batch_size=16,
        task_epochs=epochs_per_task,
        num_workers=10,
        show_progress=False,
        # This is due to random metrics not being in our objective for the NAS.
        with_random_metrics=True,
        evaluate_after_task=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Learning Trainer")

    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to the experiment directory",
    )

    parser.add_argument(
        "--model_encoding",
        type=int,
        nargs="+",
        required=True,
        help="Model to train, as encoded by the search space",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset to use",
    )

    parser.add_argument(
        "--n_tasks",
        type=int,
        default=5,
        help="Number of continual tasks",
    )

    parser.add_argument(
        "--epochs_per_task",
        type=int,
        default=1,
        help="Epochs per task",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        choices=["fixed", "expandable"],
        default="fixed",
        help="Architecture to use",
    )

    # Parse parameters and run the main loop
    args = parser.parse_args()
    main(
        model_encoding=args.model_encoding,
        experiment_dir=args.experiment_dir,
        n_tasks=args.n_tasks,
        optimiser_name="adam",
        learning_rate=7.5e-4,
        weight_decay=1e-5,
        epochs_per_task=args.epochs_per_task,
        random_seed=args.random_seed,
        dataset_name=args.dataset,
        fixed_architecture=args.architecture == "fixed",
    )
