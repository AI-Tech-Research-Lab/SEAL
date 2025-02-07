# ruff: noqa: F401
import _config

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

from functools import partial
import torch.nn as nn
import argparse
import os

# Default results directory
DEFAULT_RESULTS_DIR = "scripts-baselines/results-random"

# The only OFA family supported by the evaluator
OFA_FAMILY = "mobilenetv3"

# Default parameters for the optimiser
DEFAULT_OPTIMISER_NAME = "adam"
DEFAULT_LEARNING_RATE = 7.5e-4
DEFAULT_WEIGHT_DECAY = 1e-5


# Define the list of available trainers
TRAINERS = {
    "none": NaiveBaselineTrainer,
    "joint": JointBaselineTrainer,
    "ewc": EWCBaselineTrainer,
    "lwf": LwFBaselineTrainer,
    "si": SIBaselineTrainer,
}


def get_continual_trainer(
    trainer_name: str,
    dataset_name: str,
):
    return TRAINERS[trainer_name]


def main(
    model_encoding: str,
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

    # Get the search space and the OFA supernet
    supernet = get_ofa_evaluator(family=OFA_FAMILY, dataset=dataset_name)
    search_space = get_search_space(
        family=OFA_FAMILY,
        fixed=trainer_name != "expandable",
    )

    # Parse to int all the values to ensure the correct decoding
    sampled_architecture = search_space.decode(model_encoding)
    base_model = supernet.get_architecture_model(sampled_architecture, pretrained=True)

    # Get the experiment directory
    experiment_dir = f"{experiment_dir}@{n_tasks}/{dataset_name}"
    model_name = f"{trainer_name}-seed{random_seed}"

    # Check if the model has already been trained
    if os.path.exists(f"{experiment_dir}/models/{model_name}/history.json"):
        print(f"Model {model_name} already trained, skipping...")
        return

    # Load the trainer with the dataset
    continual_trainer = get_continual_trainer(
        dataset_name=dataset_name,
        trainer_name=trainer_name,
    )

    # Load the continual trainer
    trainer = continual_trainer(
        experiment_dir=experiment_dir,
        experiment_name=model_name,
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
        batch_size=32,
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
        "--trainer_name",
        type=str,
        choices=TRAINERS.keys(),
        default="fixed",
        help="Baseline trainer to use",
    )

    parser.add_argument(
        "--model_encoding",
        type=int,
        nargs="+",
        required=True,
        help="Model to train, as encoded by the search space",
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
        model_encoding=args.model_encoding,
        trainer_name=args.trainer_name,
        n_tasks=args.n_tasks,
        epochs_per_task=args.epochs_per_task,
        dataset_name=args.dataset_name,
        random_seed=args.random_seed,
    )
