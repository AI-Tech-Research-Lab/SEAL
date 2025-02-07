# ruff: noqa: F401
import _config

from _constants import DATASET_N_CLASSES
from continual_baselines import (
    NaiveBaselineTrainer,
    JointBaselineTrainer,
    EWCBaselineTrainer,
    SIBaselineTrainer,
    LwFBaselineTrainer,
)

from search_space import get_search_space, get_ofa_evaluator
from search_space.ofa_tools import get_net_info
from tools.metrics import binary_accuracy
from _utils import set_seed

from torchvision import models
from functools import partial
import torch.nn as nn
import argparse
import os

# Default results directory
DEFAULT_RESULTS_DIR = "scripts-baselines/results-mobilenetv3"

# Default parameters for the optimiser
DEFAULT_OPTIMISER_NAME = "adam"
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-5

# Define the list of available trainers
TRAINERS = {
    "none": NaiveBaselineTrainer,
    "joint": JointBaselineTrainer,
    "ewc": EWCBaselineTrainer,
    "lwf": LwFBaselineTrainer,
    "si": SIBaselineTrainer,
}


def main(
    model_size: str,
    trainer_name: str,
    n_tasks: int,
    epochs_per_task: int,
    dataset_name: str,
    random_seed: int,
    experiment_dir: str = DEFAULT_RESULTS_DIR,
) -> None:
    """
    Main function to initialize the trainer and start the training process.
    """
    set_seed(random_seed)

    # Get the mobilenetv3 model architecture
    if model_size == "small":
        base_model = models.mobilenet_v3_small(pretrained=True, dropout=0.0)
    elif model_size == "large":
        base_model = models.mobilenet_v3_large(pretrained=True, dropout=0.0)
    else:
        raise ValueError(f"Invalid model size: {model_size}")

    # Fix the number of classes to learn
    base_model.classifier[-1] = nn.Linear(
        base_model.classifier[-1].in_features, DATASET_N_CLASSES[dataset_name]
    )

    # Get the experiment directory
    experiment_dir = f"{experiment_dir}-{model_size}@{n_tasks}/{dataset_name}"
    model_name = f"{trainer_name}-seed{random_seed}"

    # Check if the model has already been trained
    if os.path.exists(f"{experiment_dir}/models/{model_name}/history.json"):
        print(f"Model {model_name} already trained, skipping...")
        return

    # Load the trainer with the dataset
    continual_trainer = TRAINERS[trainer_name]

    # Load the continual trainer
    trainer = continual_trainer(
        experiment_dir=experiment_dir,
        experiment_name=model_name,
        base_model=base_model,
        model_definition={"resolution": 224},
        num_tasks=n_tasks,
        random_seed=random_seed,
    )

    # Load the data and the stats to use for normalisation
    trainer.load_dataset(
        dataset_name=dataset_name,
        evaluation=True,
        normalise=True,
    )

    # Set training settings
    trainer.set_experiment_settings(
        loss_fn=nn.CrossEntropyLoss(),
        epochs_per_task=epochs_per_task,
        optim_name=DEFAULT_OPTIMISER_NAME,
        optim_params={
            "lr": DEFAULT_LEARNING_RATE,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
        },
        training_metrics={"accuracy": binary_accuracy},
        model_size_metrics={},
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
        "--model_size",
        type=str,
        choices=["small", "large"],
        default="small",
        help="Model size",
    )

    parser.add_argument(
        "--trainer_name",
        type=str,
        choices=TRAINERS.keys(),
        default="none",
        help="Baseline trainer to use",
    )

    parser.add_argument(
        "--dataset_name",
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
        default=42,
        help="Random seed",
    )

    # Parse parameters and run the main loop
    args = parser.parse_args()
    main(
        model_size=args.model_size,
        trainer_name=args.trainer_name,
        n_tasks=args.n_tasks,
        epochs_per_task=args.epochs_per_task,
        dataset_name=args.dataset_name,
        random_seed=args.random_seed,
    )
