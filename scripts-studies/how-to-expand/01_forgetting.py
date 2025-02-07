from _config import BASE_CONFIGURATIONS

from continual_learning.continual_trainers import GrowingDataContinualTrainer
from search_space import get_search_space, get_ofa_evaluator, ModelSample
from tools.metrics import binary_accuracy
from _utils import set_seed

from functools import partial
from tqdm import tqdm
from torch import nn
import argparse
import os

# Output directory where we put the results
RESULTS_DIR = "scripts-studies/how-to-expand/results-forgetting"

# Defines the number different models for each settings
N_MODELS = 5

# Define the ofa family we are using
OFA_FAMILY = "mobilenetv3"

# Define a random seed for the experiment
RANDOM_SEED = 42


def sample_architecture(
    dataset_name: str,
    ofa_family: str = OFA_FAMILY,
    random_seed: int = 42,
):
    """
    Samples an architecture from the OFA space and retrieves the model weights.
    """
    set_seed(random_seed)

    # Get the search space and its evaluator
    search_space = get_search_space(family=ofa_family, fixed=False)
    ofa_net = get_ofa_evaluator(ofa_family, dataset_name, pretrained=True)

    # Step 0: Sample an architecture making sure it is expandable
    sampled_architecture = search_space.sample(n_samples=8)[4]
    while sampled_architecture["direction"] == [0, 0, 0]:
        sampled_architecture = search_space.sample(n_samples=8)[4]

    # Retrieve the model from the OFA
    base_model = ofa_net.get_architecture_model(sampled_architecture)
    return sampled_architecture, base_model


def model_evaluator(
    distill_on_expand: bool,
    expand_is_frozen: bool,
    model: nn.Module,
    model_definition: ModelSample,
    n_tasks: int,
    epochs_per_task: int,
    dataset_name: str,
    random_seed: int,
    ofa_family: str = OFA_FAMILY,
    show_progress: bool = True,
):
    """
    Main function to initialize the trainer and start the training process.
    """
    set_seed(random_seed)

    # Experiments identifier
    identifiers = [
        "frozen" if expand_is_frozen else "NF",
        "D" if distill_on_expand else "ND",
        f"seed{random_seed}",
    ]

    # Define the results directory
    experiment_dir = os.path.join(
        RESULTS_DIR,
        f"{dataset_name}@{n_tasks}-e{epochs_per_task}",
    )
    os.makedirs(experiment_dir, exist_ok=True)

    # Step 1. Load the trainer with the dataset
    trainer = GrowingDataContinualTrainer(
        capacity_tau=100,  # This will cause a expansion on each new task
        distill_on_expand=distill_on_expand,
        weights_from_ofa=True,
        expand_is_frozen=expand_is_frozen,
        dataset_name=dataset_name,
        search_space_family=ofa_family,
        experiment_dir=experiment_dir,
        experiment_name="-".join(identifiers),
        model_definition=model_definition,
        base_model=model,
        num_tasks=n_tasks,
        random_seed=random_seed,
    )

    # Load the data and the stats to use for normalisation
    trainer.load_dataset(dataset_name=dataset_name)

    # Set training settings
    trainer.set_experiment_settings(
        loss_fn=nn.CrossEntropyLoss(),
        epochs_per_task=epochs_per_task,
        optim_name=BASE_CONFIGURATIONS["optimiser"],
        optim_params=BASE_CONFIGURATIONS["optimiser_params"],
        model_size_metrics={},
        training_metrics={"accuracy": binary_accuracy},
        augment=False,
    )

    # Step 5: Train the model (random and continual)
    trainer.train(
        task_epochs=epochs_per_task,
        show_progress=show_progress,
        with_random_metrics=True,
        evaluate_after_task=True,
        num_workers=10,
        batch_size=32,
    )


if __name__ == "__main__":
    # Define arg params for the run
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--n_tasks", type=int, default=5)
    parser.add_argument("--epochs_per_task", type=int, default=1)
    args = parser.parse_args()

    # Iterate over the models
    for i in tqdm(range(N_MODELS), desc="Evaluation", unit="model"):
        iter_sample, iter_model = sample_architecture(
            dataset_name=args.dataset,
            ofa_family="mobilenetv3",
            random_seed=(RANDOM_SEED + i),
        )

        # Define a base common evalautor
        base_evaluator = partial(
            model_evaluator,
            model=iter_model,
            model_definition=iter_sample,
            n_tasks=args.n_tasks,
            epochs_per_task=args.epochs_per_task,
            dataset_name=args.dataset,
            random_seed=(RANDOM_SEED + i),
            ofa_family=OFA_FAMILY,
            show_progress=False,
        )

        # Evaluate freezing the expansion + no distillation
        print("\n" + "-" * 90)
        print("Evaluating freezing the expansion + no distillation")
        base_evaluator(
            expand_is_frozen=True,
            distill_on_expand=False,
        )

        # Evaluate ofa initialisation, no distillation
        print("\n" + "-" * 90)
        print("Evaluating freezing the expansion + distillation")
        base_evaluator(
            expand_is_frozen=True,
            distill_on_expand=True,
        )

        # Evaluate free expansion, no distillation
        print("\n" + "-" * 90)
        print("Evaluating free expansion, no distillation")
        base_evaluator(
            expand_is_frozen=False,
            distill_on_expand=False,
        )

        # Evaluate ofa initialisation, distillation
        print("\n" + "-" * 90)
        print("Evaluating free expansion, distillation")
        base_evaluator(
            expand_is_frozen=False,
            distill_on_expand=True,
        )
