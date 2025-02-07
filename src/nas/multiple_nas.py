from surrogates import SurrogatePredictor
from nas.base_nas import BaseNAS, ModelsArchive
from nas._problems import MultiObjProblem
from nas._subsets import (
    BinaryCrossover,
    MyMutation,
    MySampling,
    SubsetProblem,
)

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from abc import abstractmethod
import numpy as np

from typing import List


class MultiObjectiveNAS(BaseNAS):
    def __init__(
        self,
        first_obj_surrogate_name: str,
        second_obj_surrogate_name: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Params for the surrogate predictors
        self.first_obj_surrogate_name = first_obj_surrogate_name
        self.second_obj_surrogate_name = second_obj_surrogate_name

        # Predefined for the kind of nas
        self.experiment_nas_results_dir = "multi-objective"

    def _subset_selection(
        self,
        candidate_metrics: np.ndarray,
        front_metrics: np.ndarray,
        K: int,
    ) -> List[int]:
        """
        Performs subset selection to diversify the Pareto front in multi-objective Neural Architecture Search (NAS).

        This method is crucial in NAS for maintaining a diverse set of architectures that represent different trade-offs
        between objectives. It helps in avoiding premature convergence and ensures a good exploration of the search space.

        ### Args:
            `candidate_metrics (np.ndarray)`: Metrics of candidate architectures.
            `front_metrics (np.ndarray)`: Metrics of the current Pareto front.
            `K (int)`: Number of candidates to select.

        ### Returns:
            `List[int]`: Indices of the selected candidates.
        """
        problem = SubsetProblem(
            candidates=candidate_metrics,
            archive=front_metrics,
            K=K,
        )

        algorithm = GA(
            pop_size=100,
            sampling=MySampling(),
            crossover=BinaryCrossover(),
            mutation=MyMutation(),
            eliminate_duplicates=True,
        )

        res = minimize(
            problem,
            algorithm,
            ("n_gen", 60),
            verbose=False,
        )

        return res.X

    def fit_surrogate_predictors(self, archive: ModelsArchive):
        """
        Fits a single surrogate model to predict only the first objective.
        """

        # Train the first objective predictor
        first_obj_predictor, first_obj_preds = self._fit_surrogate_predictor(
            archive=archive,
            surrogate_name=self.first_obj_surrogate_name,
            obj_idx=0,
        )

        second_obj_predictor, second_obj_preds = self._fit_surrogate_predictor(
            archive=archive,
            surrogate_name=self.second_obj_surrogate_name,
            obj_idx=1,
        )

        # Returns a list to comply with the interface
        return [first_obj_predictor, second_obj_predictor], [
            first_obj_preds,
            second_obj_preds,
        ]

    def generate_candidates(
        self,
        archive: ModelsArchive,
        surrogate_predictors: List[SurrogatePredictor],
        n_candidates: int,
    ):
        """
        Searches for the next K-candidates for high-fidelity evaluation of the
        lower level optimisation problem.

        > Correspond to lines [10...11] in the reference paper.
        """

        encoded_archive = [self.search_space.encode(x[0]) for x in archive]
        archive_metrics = np.column_stack(([x[1] for x in archive])).T

        # Get non-dominated architectures from archive ( top K architectures )
        front = NonDominatedSorting().do(archive_metrics, only_non_dominated_front=True)
        encoded_non_dominated = np.array(encoded_archive)[front]

        # Initialize the candidate finding optimization problem
        problem = MultiObjProblem(
            dataset_name=self.dataset_name,
            ofa_space_family=self.ofa_space_family,
            search_space=self.search_space,
            first_obj_predictor=surrogate_predictors[0],
            second_obj_predictor=surrogate_predictors[1],
            trade_off_param=0.07,
            size_metric="params",
        )

        # Define the single-level genetic algorithm to optimise
        method = NSGA2(
            pop_size=40,
            sampling=encoded_non_dominated,
            crossover=TwoPointCrossover(prob=0.9),
            mutation=PolynomialMutation(eta=1.0),
            eliminate_duplicates=True,
        )

        # Run the search to minimise the first objective
        res = minimize(
            problem,
            method,
            termination=("n_gen", 20),
            save_history=True,
            verbose=True,
        )

        # Resulting samples population
        X_population = res.pop.get("X").astype(np.int8)
        F_population = res.pop.get("F")

        # Check for duplicates in the archive
        not_duplicated = np.logical_not(
            [
                (x in [x[0] for x in archive])
                for x in [self.search_space.decode(x) for x in X_population]
            ]
        )

        # Line 11: Form a subset selection problem to short list K from pop_size
        # note that we are using the second objective to form the subset selection problem
        indices = self._subset_selection(
            candidate_metrics=F_population[not_duplicated][:, 1],
            front_metrics=archive_metrics[front, 1],
            K=n_candidates,
        )

        # From the not duplicated get the selected candidates ( note that .get("X") is sorted )
        try:
            X_candidates = X_population[not_duplicated][indices]
            candidates = [self.search_space.decode(x) for x in X_candidates]
        except TypeError:
            print("Wrong format in X_candidates")
            X_candidates = X_population[not_duplicated][indices][0]
            candidates = [self.search_space.decode(x) for x in X_candidates]

        # Predict the objective for the candidates
        first_err_preds = surrogate_predictors[0].predict(X_candidates)
        second_err_preds = surrogate_predictors[1].predict(X_candidates)

        # Note we are returning a tuple to comply with the interface
        return candidates, (first_err_preds, second_err_preds)

    @abstractmethod
    def evaluate(self, archive: ModelsArchive):
        pass
