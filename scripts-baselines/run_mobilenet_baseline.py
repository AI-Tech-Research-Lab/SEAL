from _baseline_utils import run_evaluator_script
from _utils import set_seed

import argparse


# Define the baselines methods
TRAINERS = ["none", "joint", "si", "ewc", "lwf"]

# Number of tasks we are evaluating
N_TASKS = [5, 10]

# Random aggregations ( generate a different split of the dataset )
N_RANDOM_EVALS = 2

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name.",
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

    # For each architecture, run the evaluator script
    for model_size in ["small"]:
        for n_tasks in N_TASKS:
            for trainer_name in TRAINERS:
                for i in range(N_RANDOM_EVALS):
                    print(
                        f"Running {trainer_name} on {model_size} with {n_tasks} tasks"
                    )
                    run_evaluator_script(
                        encoded_architecture=None,
                        mobilenet_size=model_size,
                        trainer_name=trainer_name,
                        dataset_name=args.dataset_name,
                        n_tasks=n_tasks,
                        epochs_per_task=args.epochs_per_task,
                        random_seed=args.random_seed + i,
                    )
