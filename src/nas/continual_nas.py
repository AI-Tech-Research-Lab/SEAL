from nas.evaluators import evaluate_fixed_model, evaluate_growing_model
from nas.single_nas import SingleObjectiveNAS
from nas.multiple_nas import MultiObjectiveNAS
from nas.base_nas import ModelSample, BaseNAS

from nas.evaluators.postprocessing import postprocess_experiment

from abc import abstractmethod
from functools import partial
from tqdm import tqdm
import numpy as np

from typing import Tuple, List


class EfficientContinualNAS(BaseNAS):
    """
    Abstract base class for Efficient Continual NAS implementations.
    Handles common logic for both single and multi-objective scenarios.
    """

    def __init__(
        self,
        n_tasks: int,
        optimiser_name: str,
        learning_rate: float,
        weight_decay: float,
        epochs_per_task: int,
        capacity_tau: float = 0.1,
        expand_is_frozen: bool = True,
        distill_on_expand: bool = True,
        weights_from_ofa: bool = True,
    ):
        # Define the parameters to run the models
        self.n_tasks = n_tasks
        self.optimiser_name = optimiser_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs_per_task = epochs_per_task

        # Parameters for the growing architecture search
        self.capacity_tau = capacity_tau
        self.expand_is_frozen = expand_is_frozen
        self.distill_on_expand = distill_on_expand
        self.weights_from_ofa = weights_from_ofa

    @abstractmethod
    def _get_model_metrics(self, experiment_dir: str) -> Tuple[float, ...]:
        """
        Abstract method to compute metrics for a model.

        Args:
            experiment_dir (str): Directory containing experiment results

        Returns:
            Tuple[float, ...]: Computed metrics (single or multiple objectives)
        """
        pass

    def evaluate(self, samples: List[ModelSample]) -> Tuple[np.ndarray, ...]:
        """
        Evaluate the archive of models.

        Args:
            samples (List[ModelSample]): List of samples to evaluate.

        Returns:
            Tuple[np.ndarray, ...]: Evaluation results, containing the objective values for each model.
        """

        _evaluate_model = partial(
            evaluate_fixed_model if self.fixed_arch else evaluate_growing_model,
            random_seed=self.random_seed,
            experiment_dir=self.experiment_dir,
            ofa_space_family=self.ofa_space_family,
            dataset_name=self.dataset_name,
            n_tasks=self.n_tasks,
            optimiser_name=self.optimiser_name,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            epochs_per_task=self.epochs_per_task,
            capacity_tau=self.capacity_tau,
            expand_is_frozen=self.expand_is_frozen,
            distill_on_expand=self.distill_on_expand,
            weights_from_ofa=self.weights_from_ofa,
        )

        encoded_samples = [self.search_space.encode(s) for s in samples]

        results_dirs = []
        for sample in tqdm(encoded_samples, desc="Evaluating models", unit="model"):
            results_dirs.append(_evaluate_model(sample))

        evaluated_archive = []
        for _sample, sample_results_dir in results_dirs:
            metrics = self._get_model_metrics(sample_results_dir)
            evaluated_archive.append(np.array(metrics))

        return tuple(evaluated_archive)


class EfficientContinualSingleNAS(EfficientContinualNAS, SingleObjectiveNAS):
    """Efficient Continual - Single Objective NAS."""

    def __init__(
        self,
        n_tasks: int,
        optimiser_name: str,
        learning_rate: float,
        weight_decay: float,
        epochs_per_task: int,
        surrogate_predictor_name: str,
        capacity_tau: float = 0.1,
        expand_is_frozen: bool = True,
        distill_on_expand: bool = True,
        weights_from_ofa: bool = True,
        **base_nas_kwargs,
    ):
        SingleObjectiveNAS.__init__(
            self, surrogate_predictor_name=surrogate_predictor_name, **base_nas_kwargs
        )
        EfficientContinualNAS.__init__(
            self,
            n_tasks=n_tasks,
            optimiser_name=optimiser_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs_per_task=epochs_per_task,
            capacity_tau=capacity_tau,
            expand_is_frozen=expand_is_frozen,
            distill_on_expand=distill_on_expand,
            weights_from_ofa=weights_from_ofa,
        )

    def _get_model_metrics(self, experiment_dir: str) -> Tuple[float]:
        """
        Compute single objective metrics for a model.

        Args:
            experiment_dir (str): Directory containing experiment results

        Returns:
            Tuple[float]: Single objective value (error rate)
        """
        _results = postprocess_experiment(
            experiment_dir, self.ofa_space_family, self.fixed_arch
        )
        objective = (1 - _results["average_accuracy"]) * 100
        return (objective,)


class EfficientContinualMultiNAS(EfficientContinualNAS, MultiObjectiveNAS):
    """Efficient Continual - Multi Objective NAS."""

    def __init__(
        self,
        n_tasks: int,
        optimiser_name: str,
        learning_rate: float,
        weight_decay: float,
        epochs_per_task: int,
        first_obj_surrogate_name: str,
        second_obj_surrogate_name: str,
        capacity_tau: float = 0.1,
        expand_is_frozen: bool = True,
        distill_on_expand: bool = True,
        weights_from_ofa: bool = True,
        **base_nas_kwargs,
    ):
        MultiObjectiveNAS.__init__(
            self,
            first_obj_surrogate_name=first_obj_surrogate_name,
            second_obj_surrogate_name=second_obj_surrogate_name,
            **base_nas_kwargs,
        )
        EfficientContinualNAS.__init__(
            self,
            n_tasks=n_tasks,
            optimiser_name=optimiser_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs_per_task=epochs_per_task,
            capacity_tau=capacity_tau,
            expand_is_frozen=expand_is_frozen,
            distill_on_expand=distill_on_expand,
            weights_from_ofa=weights_from_ofa,
        )

    def _get_model_metrics(self, experiment_dir: str) -> Tuple[float, float]:
        """
        Compute multi-objective metrics for a model.

        Args:
            experiment_dir (str): Directory containing experiment results

        Returns:
            Tuple[float, float]: Multiple objective values (error rate, flatness)
        """
        _results = postprocess_experiment(
            experiment_dir, self.ofa_space_family, self.fixed_arch
        )
        first_objective = (1 - _results["average_accuracy"]) * 100
        second_objective = (1 - _results["average_flatness"]) * 100
        return (first_objective, second_objective)
