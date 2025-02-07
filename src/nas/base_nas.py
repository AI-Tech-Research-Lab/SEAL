from surrogates import get_surrogate_predictor, SurrogatePredictor
from _utils import convert_to_json_serializable, set_seed
from search_space import get_search_space, ModelSample
from tools.stats import get_correlation

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from abc import ABC, abstractmethod
import numpy as np
import json
import os
import re


from typing import Dict, List, Optional, Tuple, TypeAlias


ModelsArchive: TypeAlias = List[Tuple[ModelSample, np.ndarray]]


class BaseNAS(ABC):
    def __init__(
        self,
        experiment_dirname: str,
        ofa_space_family: str,
        fixed_arch: bool,
        dataset_name: str,
        n_initial_samples: int,
        search_iterations: int,
        archs_per_iter: int,
        resume_search: bool,
        resume_from_iter: Optional[int],
        random_seed: int,
    ):
        self.ofa_space_family = ofa_space_family
        self.dataset_name = dataset_name

        # Define the search space
        self.search_space = get_search_space(family=ofa_space_family, fixed=fixed_arch)
        self.fixed_arch = fixed_arch

        # NAS search parameters
        self.n_initial_samples = n_initial_samples
        self.archs_per_iter = archs_per_iter

        # > Used to track the number of iterations performed
        self.search_iterations = search_iterations
        self.initial_search_it = 0

        # Whether to continue searching from a previous checkpoint
        self.resume_search = resume_search
        self.resume_from_iter = resume_from_iter

        # Sets the nadir point for the hypervolume
        self.nadir_point: np.ndarray

        # Create an nas-experiment dir to store the results
        self.experiment_dir = experiment_dirname
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Further specifies the type of search we are using
        self.experiment_nas_results_dir: str

        # Set the seed for reproducibility ( this will be used for sampling )
        self.random_seed = random_seed
        set_seed(random_seed)

    def _store_archive_to_dir(self, archive: ModelsArchive, it: int) -> bool:
        """
        Stores the archive to the experiment directory, in the `archives` subdirectory.

        ### Args:
            `archive (ModelsArchive)`: Archive of evaluated models.
            `it (int)`: Iteration number.

        ### Returns:
            `bool`: Whether the operation was successful.
        """

        # Builds the filepath and make sure it exists
        filepath = os.path.join(
            self.experiment_dir,
            self.experiment_nas_results_dir,
            "archives",
            f"iter_{it}.archive",
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Format the archive for better legibility
        formatted_archive = []
        for sample, metrics in archive:
            formatted_archive.append(
                {
                    "sample": sample,
                    "metrics": metrics,
                }
            )

        # Store the archive as a json file
        serializable_archive = convert_to_json_serializable(formatted_archive)
        with open(filepath, "w") as handle:
            json.dump(serializable_archive, handle, indent=4)

        return True

    def _load_archive_from_dir(
        self, it: Optional[int] = None
    ) -> Tuple[ModelsArchive, int]:
        """
        Tries to load the experiment data from the directory. If a specific iteration is
        provided, only looks for the data related to such iteration.

        ### Args:
            `it (int)`: The iteration number to load.

        ### Returns:
            `Tuple[ModelsArchive, int]`: A tuple containing the loaded archive and the
            corresponding iteration number.
        """

        # Get all the .archives files
        archives_dir = os.path.join(
            self.experiment_dir,
            self.experiment_nas_results_dir,
            "archives",
        )
        checkpoint_files = [f for f in os.listdir(archives_dir) if ".archive" in f]

        # Sort the archives in ascending order
        checkpoint_files.sort(key=lambda name: int(re.search(r"\d+", name).group()))
        last_checkpoint = checkpoint_files[-1]

        # Access the requested checkpoint if available
        if it is not None:
            if f"iter_{it}.archive" not in checkpoint_files:
                raise FileNotFoundError(f".archive file for iteration {it} not found")
            last_checkpoint = f"iter_{it}.archive"

        # Get the loaded iteration
        loaded_it = int(re.search(r"\d+", last_checkpoint).group())

        # Load the last checkpoint
        filepath = os.path.join(archives_dir, last_checkpoint)
        with open(filepath, "r") as handle:
            loaded_archive = json.load(handle)

        # Read from the stored format
        archive = [
            (sample["sample"], np.array(sample["metrics"])) for sample in loaded_archive
        ]
        return archive, loaded_it

    def _fit_surrogate_predictor(
        self,
        archive: ModelsArchive,
        surrogate_name: str,
        obj_idx: int,
    ) -> Tuple[SurrogatePredictor, np.ndarray]:
        """
        Fits a surrogate model to predict the desired metric from the
        archive dataset.

        ### Args:
            `archive (ModelsArchive)`: Archive of evaluated models.
            `surrogate_name (str)`: Name of the predictor to fit.
            `obj_idx (int)`: Index of the objective to fit the surrogate model.

        ### Returns:
            `Tuple[Any, np.ndarray]`: A tuple containing the surrogate model and its
            predictions.

        """

        # Get the inputs and targets for the surrogate model
        encoded_archs = np.array([self.search_space.encode(x[0]) for x in archive])
        obj_targets = np.array([x[1][obj_idx] for x in archive])

        # Fit the surrogate model
        _predictor = get_surrogate_predictor(
            model_name=surrogate_name,
            inputs=encoded_archs,
            targets=obj_targets,
            random_seed=self.random_seed,
        )

        # Return the surrogate model and its predictions
        return _predictor, _predictor.predict(encoded_archs)

    def _compute_hypervolume(self, archive: ModelsArchive) -> float:
        """
        Computes the hypervolume of the current archive, normalised by the reference point.

        The hypervolume is a metric used in multi-objective optimization to measure the volume of the
        objective space dominated by a set of non-dominated solutions. It provides a quantitative
        measure of the quality and diversity of the Pareto front approximation.

        In the context of Neural Architecture Search (NAS), the hypervolume is particularly important
        because:
        1. It captures both convergence and diversity of the found architectures.
        2. It allows for comparison between different runs or algorithms.
        3. It provides a single scalar value to track the progress of the search.
        4. It is sensitive to improvements in any objective, making it suitable for many-objective problems.

        The hypervolume is normalized by the reference point to make it comparable across different
        scales and problem instances.

        ### Args:
            `archive (ModelsArchive)`: The current archive of models.

        ### Returns:
            `Dict[str, float]`: The hypervolume of the current archive, normalised by the reference point.
        """

        # Get the metrics from the iterations archive
        _metrics = np.array([x[1] for x in archive])

        # Computes the non-dominated front of the current archive
        _front = NonDominatedSorting().do(_metrics, only_non_dominated_front=True)
        non_dominated_archive = _metrics[_front, :]

        # Compute the reference point for the hypervolume calculation
        ref_point = 1.01 * self.nadir_point
        hv_index = HV(ref_point=ref_point)

        # Compute the hypervolume and normalise it if required
        return hv_index(non_dominated_archive) / np.prod(ref_point)

    def load_models_archive(self) -> ModelsArchive:
        """
        Handles the loading/initialisation of a models archive. There are branch behaviours for
        this function:

        1) Starting from scratch - Random sampling and evaluation is performed.
        2) Resuming a previous search - The search is resumed from the last saved iteration.

        We assume the second behaviour when the `self.save_dir` path contains valid data.
        """

        if self.resume_search:
            archive, retrieved_it = self._load_archive_from_dir(
                it=self.resume_from_iter,
            )
        else:
            retrieved_it = 0

            # Sample the architectures randomly
            _samples = self.search_space.get_initial_samples(self.n_initial_samples)
            _metrics = self.evaluate(_samples)

            # Group the decoded sample and the metrics to create the archive.
            archive = [entry for entry in zip(_samples, _metrics)]

            # Store the initial archive
            self._store_archive_to_dir(archive=archive, it=retrieved_it)

        # Compute the nadir point to keep track of the hypervolume (only in multiple obj)
        archive_metrics = np.array([x[1] for x in archive])
        self.nadir_point = np.max(archive_metrics, axis=0)

        # Update the initial search iteration counter
        self.initial_search_it = retrieved_it

        return archive

    def store_iteration_metrics(
        self,
        candidate_archs: List[ModelSample],
        candidate_metrics: Tuple[np.ndarray, ...],
        corr_metrics: Dict[str, np.ndarray],
        extra_metrics: Dict[str, float | np.ndarray],
        it: int,
    ):
        """
        Stores the iteration statistics in a JSON file and updated a `pd.DataFrame` to access
        the data in memory.
        """
        assert len(candidate_archs) == len(candidate_metrics), "Invalid metrics length"

        # Encode the archs to be able to store them
        encoded_archs = [self.search_space.encode(x) for x in candidate_archs]
        str_encoded_archs = ["".join([str(x) for x in arch]) for arch in encoded_archs]

        # Group the arch metrics with their corresponding metrics and add the correlations
        store_data = {
            "candidate_metrics": {
                str_encoded_archs[i]: candidate_metrics[i]
                for i in range(len(candidate_archs))
            },
            "surrogate_metrics": corr_metrics,
            **extra_metrics,
        }

        # Create the dir for the .json file at the base of the experiment.
        filepath = os.path.join(
            self.experiment_dir,
            self.experiment_nas_results_dir,
            "metrics",
            f"iter_{it}.metrics",
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Serialise and store the data
        serializable_data = convert_to_json_serializable(store_data)
        with open(filepath, "w+") as handle:
            json.dump(serializable_data, handle)

    @abstractmethod
    def fit_surrogate_predictors(
        self, archive: ModelsArchive
    ) -> Tuple[List[SurrogatePredictor], List[np.ndarray]]:
        """
        Fits the surrogates models for the metrics defined in the NAS scope. Internally,
        calls the function `_fit_surrogate_predictor` for each metric.

        ### Args:
            `archive (ModelsArchive)`: Archive of evaluated models.

        ### Returns:
            `Tuple[List[SurrogatePredictor], List[np.ndarray]]`: A tuple containing the list of
            surrogate models and their predictions.
        """
        pass

    @abstractmethod
    def generate_candidates(
        self,
        archive: ModelsArchive,
        surrogate_predictors: List[SurrogatePredictor],
        n_candidates: int,
    ) -> Tuple[List[ModelSample], Tuple[np.ndarray]]:
        """
        Generate candidate architectures using genetic algorithm optimization.

        ### Args:
            `archive (ModelsArchive)`: Archive of evaluated models.
            `surrogate_predictors   (s)`: Surrogate objective predictor model(s).
            `n_candidates (int)`: Number of candidates to generate.

        Returns:
            `Tuple[List[str], np.ndarray]`: A tuple containing the list of candidate
            architectures and their predicted objective values.
        """
        pass

    @abstractmethod
    def evaluate(self, archive: List[ModelSample]) -> Tuple[np.ndarray, ...]:
        """
        Abstract method to evaluate the archive of models.

        ### Args:
            `archive (List[ModelSample])`: Archive of models to evaluate.

        ### Returns:
            `Tuple[np.ndarray, ...]`: Evaluation results, each element of the tuple is the result
            for a different model.
        """
        pass

    def compute_correlation_metrics(
        self,
        archive_metrics: Tuple[np.ndarray, ...],
        candidate_metrics: Tuple[np.ndarray, ...],
        archive_err_preds: Tuple[np.ndarray, ...],
        candidates_err_preds: Tuple[np.ndarray, ...],
    ) -> Dict[str, np.ndarray]:
        """
        Compute the correlation between the surrogate predictions for the archive and candidates
        sets with the actually evaluated metrics.
        """

        # Aggregate the data structures
        evaluated_metrics = np.vstack((archive_metrics, candidate_metrics))

        # Prepare predicted errors shape
        _archive_err_preds = np.column_stack([*archive_err_preds])
        _candidates_err_preds = np.column_stack([*candidates_err_preds])
        predicted_metrics = np.vstack((_archive_err_preds, _candidates_err_preds))

        # Compute the correlation metrics
        rmse, rho, tau = [], [], []
        for i in range(evaluated_metrics.shape[1]):
            rmse_i, rho_i, tau_i = get_correlation(
                predicted_metrics[:, i], evaluated_metrics[:, i]
            )
            rmse.append(rmse_i)
            rho.append(rho_i)
            tau.append(tau_i)

        return {
            "rmse": np.array(rmse) if len(rmse) > 1 else rmse[0],
            "rho": np.array(rho) if len(rho) > 1 else rho[0],
            "tau": np.array(tau) if len(tau) > 1 else tau[0],
        }

    def compute_extra_metrics(self, archive: ModelsArchive) -> Dict[str, float]:
        """
        Computes any other required or logging metrics to track during the NAS search.
        """
        return {"hv": self._compute_hypervolume(archive=archive)}

    def search(self):
        """
        Provides the template for the entire NAS pipeline and the inner steps define
        in the original paper.
        """
        models_archive = self.load_models_archive()

        for it in range(self.initial_search_it + 1, self.search_iterations + 1):
            # Line 9: Fit the surrogate models for the archive set.
            print(f"\n> Fitting surrogate models for iteration {it}")
            obj_predictors, archive_err_preds = self.fit_surrogate_predictors(
                archive=models_archive
            )

            # Lines 10-11: Generate subset of candidate architectures.
            print(f"\n> Generating candidate models for iteration {it}")
            candidates, candidates_err_preds = self.generate_candidates(
                archive=models_archive,
                surrogate_predictors=obj_predictors,
                n_candidates=self.archs_per_iter,
            )

            # Lines 12-14: Evaluate the candidate architectures.
            print(f"\n> Evaluating candidate models for iteration {it}")
            candidate_metrics = self.evaluate(candidates)

            # Opt: Compute correlation metrics between the predicted and true objectives.
            corr_metrics = self.compute_correlation_metrics(
                archive_metrics=tuple([sample[1] for sample in models_archive]),
                candidate_metrics=candidate_metrics,
                archive_err_preds=archive_err_preds,
                candidates_err_preds=candidates_err_preds,
            )

            # Line 15: Updatings the model's archive.
            candidate_archive = [entry for entry in zip(candidates, candidate_metrics)]
            models_archive.extend(candidate_archive)

            # Opt: Compute any other metrics
            extra_metrics = self.compute_extra_metrics(archive=models_archive)

            # Opt: Store the updated archive
            self._store_archive_to_dir(archive=models_archive, it=it)

            # Opt: Store the iteration information.
            self.store_iteration_metrics(
                candidate_archs=candidates,
                candidate_metrics=candidate_metrics,
                corr_metrics=corr_metrics,
                extra_metrics=extra_metrics,
                it=it,
            )
