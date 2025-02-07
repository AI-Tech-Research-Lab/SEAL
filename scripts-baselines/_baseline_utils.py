from _config import OFA_FAMILY

from search_space import get_search_space, ModelSample

import subprocess

from typing import List, Optional


def sample_random_architectures(n_samples: int = 1) -> List[ModelSample]:
    """
    Sample a random number of architectures from the search space.
    """
    search_space = get_search_space(family=OFA_FAMILY, fixed=True)
    sampled_architectures = search_space.sample(n_samples=n_samples)
    return [search_space.encode(arch) for arch in sampled_architectures]


def run_evaluator_script(
    trainer_name: str,
    dataset_name: str,
    n_tasks: int,
    epochs_per_task: int,
    random_seed: int,
    encoded_architecture: Optional[List[int]] = None,
    mobilenet_size: Optional[str] = None,
) -> None:
    """
    Run the evaluator script for a given encoded architecture.
    """
    assert encoded_architecture is not None or mobilenet_size is not None

    SCRIPT_PATH = (
        "scripts-baselines/00_model_evaluator.py"
        if mobilenet_size is None
        else "scripts-baselines/01_mobilenetv3_evaluator.py"
    )

    # Model params
    model_params = []
    if encoded_architecture is not None:
        model_params.extend(map(str, encoded_architecture))
    else:
        model_params = [mobilenet_size]

    script_params = [
        "--trainer_name",
        trainer_name,
        "--model_encoding" if encoded_architecture is not None else "--model_size",
        *model_params,
        "--dataset_name",
        dataset_name,
        "--n_tasks",
        str(n_tasks),
        "--epochs_per_task",
        str(epochs_per_task),
        "--random_seed",
        str(random_seed),
    ]

    subprocess.run(
        [
            "python",
            SCRIPT_PATH,
            *script_params,
        ],
        check=True,
    )
