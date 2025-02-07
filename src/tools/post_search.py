from search_space import get_ofa_evaluator, get_search_space, ModelSample
from search_space.ofa_tools import get_net_info

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.decision_making import (
    DecisionMaking,
    find_outliers_upper_tail,
    NeighborFinder,
)

import numpy as np
import json
import os
import re


class HighTradeoffPoints(DecisionMaking):
    def __init__(self, epsilon=0.125, n_survive=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive  # number of points to be selected

    def _do(self, F, **kwargs):
        n, m = F.shape

        neighbors_finder = NeighborFinder(
            F,
            epsilon=0.125,
            n_min_neigbors="auto",
            consider_2d=False,
        )

        mu = np.full(n, -np.inf)

        # for each solution in the set calculate the least amount of improvement per unit deterioration
        for i in range(n):
            # for each neighbour in a specific radius of that solution
            neighbors = neighbors_finder.find(i)

            # calculate the trade-off to all neighbours
            diff = F[neighbors] - F[i]

            # calculate sacrifice and gain
            sacrifice = np.maximum(0, diff).sum(axis=1)
            gain = np.maximum(0, -diff).sum(axis=1)

            # np.warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)
        if self.n_survive is not None:
            return np.argsort(mu)[-self.n_survive :]
        else:
            return find_outliers_upper_tail(
                mu
            )  # return points with trade-off > 2*sigma


def get_best_archive(
    base_dir: str,
    dataset: str,
    model_type: str,
    seed: int,
    nas_type: str = "multi-objective",
):
    """
    Given an experiment identification, reads the archive and retrives the last
    one available.
    """

    # Form a path to the archive from the specifications
    experiment_archive_dir = os.path.join(
        base_dir,
        f"efficient-{dataset}",
        f"{model_type}-seed{seed}",
        nas_type,
    )

    # Get the last available archive
    experiment_archives = os.listdir(os.path.join(experiment_archive_dir, "archives"))
    experiment_archives.sort(key=lambda name: int(re.search(r"\d+", name).group()))
    last_archive = experiment_archives[-1]

    # Read the archive data as a .json file
    archive_path = os.path.join(experiment_archive_dir, "archives", last_archive)
    archive_data = json.load(open(archive_path))
    return archive_data


def get_archive_best_models(archive_models, archive_metrics, n_best=None):
    """
    Get the best models from the archive based on the non-dominated front
    and the trade-off points.
    """

    if archive_metrics.shape[1] > 1:
        # Get the non-dominated front
        front = NonDominatedSorting().do(archive_metrics, only_non_dominated_front=True)

        # Get non-dominated models and metrics
        non_dominated_models = np.array(archive_models)[front]
        non_dominated_metrics = archive_metrics[front, :]

        # choose the architectures with highest trade-off
        dm = HighTradeoffPoints(n_survive=n_best)
        best_model_indices = dm.do(non_dominated_metrics)

        # Format back the metrics to the original non-error format
        best_metrics = non_dominated_metrics[best_model_indices, :]
        best_models = non_dominated_models[best_model_indices]

    else:
        # Sort the archive metrics by the single objective
        front = np.argsort(archive_metrics, axis=0)
        best_model_indices = front[:n_best]

        # Get the models and metrics
        best_metrics = archive_metrics[best_model_indices, 0]
        best_models = np.array(archive_models)[best_model_indices][:, 0]

    return best_models, 100 - best_metrics


def get_base_model_size(dataset: str, model_sample: ModelSample):
    """
    Utiliy function used to acquirosthe model size from a model sample.
    """

    # Get the ofa evaluator for the model
    ofa_evaluator = get_ofa_evaluator(family="mobilenetv3", dataset=dataset)

    # Build the model definition
    architecture_model = ofa_evaluator.get_architecture_model(model_sample)
    model_resolution = model_sample["resolution"]

    # Get the size info from the model
    net_info = get_net_info(
        architecture_model,
        input_shape=(3, model_resolution, model_resolution),
    )

    return net_info["params"]


def get_model_flatness(
    base_dir: str,
    dataset: str,
    model_type: str,
    seed: int,
    model_sample: ModelSample,
):
    """
    Function that reads the model's flatness in single-objective problems. This is
    required as the flatness is not stored in single-obj search.
    """
    # Build the experiments dir
    experiment_models_dir = os.path.join(
        base_dir,
        f"efficient-{dataset}",
        f"{model_type}-seed{seed}",
        "models",
    )

    # Get the encoded models
    search_space = get_search_space(family="mobilenetv3", fixed=model_type == "fixed")
    model_encoding = search_space.encode(model_sample)
    model_name = "".join(map(str, model_encoding)) + f"-{seed}"

    # Read the model data
    model_dir = os.path.join(experiment_models_dir, model_name)
    history_path = os.path.join(model_dir, "history.json")
    model_history = json.load(open(history_path))

    # Get the flatness
    final_training_metrics = model_history["training_metrics"]["validation"]
    flatness = [task["flatness"][-1] for task in final_training_metrics.values()]
    mean_flatness = 100 * np.mean(flatness)

    return mean_flatness
