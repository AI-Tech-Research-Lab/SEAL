from _constants import SINGLE_OBJ_TRADE_OFF_PARAM
from nas.base_nas import BaseNAS, ModelsArchive
from nas._problems import SingleObjProblem
from surrogates import SurrogatePredictor

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize

from abc import abstractmethod
import numpy as np

from typing import List


class SingleObjectiveNAS(BaseNAS):
    """
    NAS for single objective optimisation.
    """

    def __init__(self, surrogate_predictor_name: str, **base_kwargs):
        super().__init__(**base_kwargs)

        # Defines the name of the surrogate predictor to use
        self.surrogate_predictor_name = surrogate_predictor_name

        # Predefined for the kind of nas
        self.experiment_nas_results_dir = "single-objective"

    def fit_surrogate_predictors(self, archive: ModelsArchive):
        """
        Fits a single surrogate model to predict only the first objective.
        """
        # Single objective optimisation
        surrogate_predictor, surrogate_predictions = self._fit_surrogate_predictor(
            archive=archive,
            surrogate_name=self.surrogate_predictor_name,
            obj_idx=0,
        )

        # Returns a list to comply with the interface
        return [surrogate_predictor], [surrogate_predictions]

    def generate_candidates(
        self,
        archive: ModelsArchive,
        surrogate_predictors: List[SurrogatePredictor],
        n_candidates: int,
    ):
        """
        Generates the candidate architectures using the pre-defined genetic algorithm.
        > Note: the metrics are always associated to the "error" as the algorithm seeks to
        minimise the objective.
        """
        # Sorts the archive by the first objective
        archive.sort(key=lambda x: x[1][0])

        # Extract the top K subnets with the best metrics to init the population
        K_biased_pop = 40
        top_K_subnets = np.array(
            [self.search_space.encode(x[0]) for x in archive[:K_biased_pop]]
        )

        # Define the problem to optimise based on the first objective
        problem = SingleObjProblem(
            dataset_name=self.dataset_name,
            ofa_space_family=self.ofa_space_family,
            search_space=self.search_space,
            obj_predictor=surrogate_predictors[0],
            trade_off_param=SINGLE_OBJ_TRADE_OFF_PARAM,
            size_metric="params",
        )

        # Define the genetic algorithm to optimise
        # > This method consistently outperforms the GA algorithm
        method = DE(
            pop_size=K_biased_pop,
            sampling=top_K_subnets,
            CR=0.9,
            F=0.8,
            variant="DE/rand/1/bin",  # Outperforms the DE/rand/1/bin
            dither="vector",
            jitter=False,
        )

        # Runs the search to minimise the first objective
        res = minimize(
            problem,
            method,
            termination=("n_gen", 45),  # Usually enough to reduce to 1e-7
            save_history=True,
            verbose=True,
            seed=self.random_seed,
        )

        # Check for duplicates in the archive
        X_candidates = res.pop.get("X").astype(np.int8)

        not_duplicated = np.logical_not(
            [
                x in [x[0] for x in archive]
                for x in [self.search_space.decode(x) for x in X_candidates]
            ]
        )

        # From the not duplicated keep only the top K archs ( note that .get("X") is sorted )
        candidates_pop = X_candidates[not_duplicated][:n_candidates]
        candidates = [self.search_space.decode(x) for x in candidates_pop]

        # Note we are returning a tuple to comply with the interface
        return candidates, (surrogate_predictors[0].predict(candidates_pop),)

    @abstractmethod
    def evaluate(self, archive: ModelsArchive):
        pass
