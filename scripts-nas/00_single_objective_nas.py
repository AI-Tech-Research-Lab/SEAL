from _config import CONFIGS_DIR, NAS_RESULTS_DIR

from nas.continual_nas import EfficientContinualSingleNAS
import argparse
import shutil
import yaml
import os

from typing import Dict, Any


# Define the experiments type ( constant )
EXPERIMENT_NAME = "efficient"


def load_config(config_filepath: str) -> Dict[str, Any]:
    """
    Load the YAML configuration file associated with the dataset.

    Args:
        config_filepath (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration parameters.
    """
    with open(config_filepath, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Learning Trainer")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        required=True,
        help="Random seed",
    )

    parser.add_argument(
        "--fixed",
        type=bool,
        required=False,
        help="Whether to use a fixed architecture",
    )
    args = parser.parse_args()

    # Make the experiment directory
    experiment_dir = os.path.join(
        NAS_RESULTS_DIR,
        f"{EXPERIMENT_NAME}-{args.dataset}",
        f"{'fixed' if args.fixed else 'growing'}-seed{args.random_seed}",
    )
    os.makedirs(experiment_dir, exist_ok=True)

    # Retrive the configuaritions file for the dataset
    config_filepath = f"{CONFIGS_DIR}/{args.dataset}.yml"
    config = load_config(config_filepath)

    # Copy the config file to the experiment directory
    single_objective_dir = os.path.join(experiment_dir, "single-objective")
    os.makedirs(single_objective_dir, exist_ok=True)
    shutil.copy(config_filepath, os.path.join(single_objective_dir, "config.yaml"))

    # Parse parameters and run the main loop
    nas = EfficientContinualSingleNAS(
        n_tasks=config["continual"]["n_tasks"],
        optimiser_name=config["optimiser"]["optimiser_name"],
        learning_rate=config["optimiser"]["learning_rate"],
        weight_decay=config["optimiser"]["weight_decay"],
        epochs_per_task=config["continual"]["epochs_per_task"],
        surrogate_predictor_name=config["surrogate_models"]["first-obj"],
        fixed_arch=args.fixed,
        capacity_tau=config["growing"]["capacity_tau"],
        expand_is_frozen=config["growing"]["expand_is_frozen"],
        distill_on_expand=config["growing"]["distill_on_expand"],
        weights_from_ofa=config["growing"]["weights_from_ofa"],
        dataset_name=args.dataset,
        experiment_dirname=experiment_dir,
        ofa_space_family=config["search_space"],
        n_initial_samples=config["nas"]["n_initial_samples"],
        search_iterations=config["nas"]["search_iterations"],
        archs_per_iter=config["nas"]["archs_per_iter"],
        resume_search=config["nas"]["resume_search"],
        resume_from_iter=config["nas"]["resume_from_iter"],
        random_seed=args.random_seed,
    )

    # Run the search
    nas.search()
