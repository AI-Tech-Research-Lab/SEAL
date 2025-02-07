import subprocess
import argparse

RUN_BASELINE_SCRIPT = "scripts-baselines/run_baseline.py"


# Define the available list of baselines
TRAINERS = ["none", "joint", "ewc", "si", "lwf"]

# Define the number of tasks we are using
N_TASKS = [5, 10]


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name.",
    )

    args.add_argument(
        "--random_seed",
        type=int,
        required=True,
        help="Defines the random seed for the experiment.",
    )

    args = args.parse_args()

    # For each architecture, run the evaluator script
    for n_tasks in N_TASKS:
        for trainer in TRAINERS:
            print(f"Running {trainer} for {n_tasks} tasks")

            # Define the trainer script params
            trainer_script_params = [
                "--trainer_name",
                trainer,
                "--dataset_name",
                args.dataset_name,
                "--n_tasks",
                str(n_tasks),
                "--epochs_per_task",
                str(1),
                "--random_seed",
                str(args.random_seed),
            ]

            # Run the trainer script
            subprocess.run(
                [
                    "python",
                    RUN_BASELINE_SCRIPT,
                    *trainer_script_params,
                ]
            )
