from _baseline_utils import sample_random_architectures, run_evaluator_script
from _utils import set_seed

import argparse

EVALUATOR_SCRIPT = "scripts-baselines/00_model_evaluator.py"


TRAINERS = ["none", "joint", "si", "ewc", "lwf"]

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument(
        "--trainer_name",
        type=str,
        required=True,
        choices=TRAINERS,
        help="Trainer name.",
    )

    args.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name.",
    )

    args.add_argument(
        "--n_tasks",
        type=int,
        required=True,
        help="Number of tasks.",
    )

    args.add_argument(
        "--epochs_per_task",
        type=int,
        required=True,
        help="Number of epochs per task.",
    )

    args.add_argument(
        "--random_seed",
        type=int,
        required=True,
        help="Defines the random seed for the experiment.",
    )

    args = args.parse_args()

    # Set random seed for reproducibility
    set_seed(args.random_seed)

    # Sample the architectures
    encoded_architectures = sample_random_architectures(n_samples=5)

    # For each architecture, run the evaluator script
    for i, encoded_architecture in enumerate(encoded_architectures):
        run_evaluator_script(
            encoded_architecture=encoded_architecture,
            trainer_name=args.trainer_name,
            dataset_name=args.dataset_name,
            n_tasks=args.n_tasks,
            epochs_per_task=args.epochs_per_task,
            random_seed=args.random_seed + i,
        )
