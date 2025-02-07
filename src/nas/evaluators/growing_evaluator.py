from .config import BATCH_SIZE, NUM_WORKERS

from continual_learning.continual_trainers import GrowingDataContinualTrainer
from search_space import get_search_space, get_ofa_evaluator
from search_space.ofa_tools import get_net_info
from tools.metrics import binary_accuracy
from _utils import set_seed

import torch.nn as nn
import argparse
import os


def main(
    experiment_dir: str,
    model_encoding: str,
    search_space_family: str,
    dataset: str,
    n_tasks: int,
    optimiser_name: str,
    learning_rate: float,
    weight_decay: float,
    epochs_per_task: int,
    capacity_tau: float,
    expand_is_frozen: bool = False,
    distill_on_expand: bool = True,
    weights_from_ofa: bool = True,
    random_seed: int = 42,
):
    """
    Main function to initialize the trainer and start the training process.
    """
    set_seed(random_seed)

    # Get the search space and the OFA supernet
    search_space = get_search_space(family=search_space_family, fixed=False)
    supernet = get_ofa_evaluator(family=search_space_family, dataset=dataset)

    # Parse to int all the values to ensure the correct decoding
    sampled_architecture = search_space.decode(model_encoding)
    base_model = supernet.get_architecture_model(sampled_architecture)

    # Get the model size-related metrics
    model_resolution = sampled_architecture["resolution"]
    model_size_metrics = get_net_info(
        base_model,
        input_shape=(3, model_resolution, model_resolution),
    )

    # Check if the model has already been trained
    model_name = "".join(map(str, model_encoding))
    experiment_name = f"{model_name}-{random_seed}"
    results_dir = os.path.join(experiment_dir, "models", experiment_name)
    if os.path.isfile(os.path.join(results_dir, "history.json")):
        print(f"\nExperiment {experiment_name} already exists, skipping")
        return

    # Load the trainer with the dataset
    trainer = GrowingDataContinualTrainer(
        capacity_tau=capacity_tau,
        expand_is_frozen=expand_is_frozen,
        distill_on_expand=distill_on_expand,
        weights_from_ofa=weights_from_ofa,
        dataset_name=dataset,
        search_space_family=search_space_family,
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        model_definition=sampled_architecture,
        base_model=base_model,
        num_tasks=n_tasks,
        random_seed=random_seed,
    )

    # Load the data and the stats to use for normalisation
    trainer.load_dataset(dataset_name=dataset)

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
        batch_size=BATCH_SIZE,
        task_epochs=epochs_per_task,
        num_workers=NUM_WORKERS,
        show_progress=False,
        # This is due to random metrics not being in our objective for the NAS.
        with_random_metrics=False,
        evaluate_after_task=False,
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
        "--search_space_family",
        type=str,
        default="mobilenetv3",
        help="Search space to use",
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
        "--optimiser_name",
        type=str,
        default="adam",
        help="Optimizer name",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Weight decay",
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

    # Parse parameters and run the main loop
    args = parser.parse_args()
    main(
        experiment_dir=args.experiment_dir,
        model_encoding=args.model_encoding,
        search_space_family=args.search_space_family,
        dataset=args.dataset,
        n_tasks=args.n_tasks,
        optimiser_name=args.optimiser_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs_per_task=args.epochs_per_task,
        random_seed=args.random_seed,
    )
