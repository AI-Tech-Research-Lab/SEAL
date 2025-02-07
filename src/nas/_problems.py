from _constants import SINGLE_OBJ_TRADE_OFF_PARAM

from search_space import get_ofa_evaluator, FixedSearchSpace, GrowthSearchSpace
from search_space.ofa_tools import get_net_info
from surrogates import SurrogatePredictor

from pymoo.core.problem import Problem
import numpy as np


class SingleObjProblem(Problem):
    def __init__(
        self,
        dataset_name: str,
        ofa_space_family: str,
        search_space: FixedSearchSpace | GrowthSearchSpace,
        obj_predictor: SurrogatePredictor,
        trade_off_param: float = SINGLE_OBJ_TRADE_OFF_PARAM,
        size_metric: str = "params",
    ):
        """
        The optimisation problem for finding the next candidates architectures with
        a single objective metric.

        This problems takes the same objective as the EfficientNet paper. A weighting
        between the accuracy of the model and its size.

        Note: the lower and upper bounds comes from the encoding space with are using
        in the search space (bit strings).

        ### Args:
            `search_space (BaseOFASearchSpace)`: The search space of the problem.
            `obj_predictor (SurrogatePredictor)`: The surrogate model used to predict the objective metric.
            `dataset_name (str)`: The name of the dataset to use for the evaluation.
        """
        super().__init__(
            n_var=search_space.nvar,
            n_obj=1,
            n_ieq_constr=1 if isinstance(search_space, GrowthSearchSpace) else 0,
            n_eq_constr=0,
            type_var=np.int8,
        )

        # Define the search space to decode the models
        self.search_space = search_space

        # Define the engine to access the models
        self.engine = get_ofa_evaluator(
            family=ofa_space_family,
            dataset=dataset_name,
        )

        # Define the parameters for the objective function
        self.obj_predictor = obj_predictor
        self.trade_off_param = trade_off_param
        self.size_metric = size_metric

        # Bounds for the generated X samples (encoded into bit strings)
        self.xl = np.zeros(self.n_var, dtype=np.int8)
        self.xu = 2 * np.ones(self.n_var, dtype=np.int8)

        # Define an upper bound for the resolution
        _res_upper_bound = int(len(self.search_space.input_resolutions) - 1)

        # Check whether we are using expanding architectures
        if isinstance(self.search_space, GrowthSearchSpace):
            self.xu[-1 - self.search_space.n_grow_bits] = _res_upper_bound
        else:
            self.xu[-1] = _res_upper_bound

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        """
        Evaluates the proposed samples using the surrogate model.

        ### Args:
            `x (np.ndarray)`: The proposed candidate architectures. Shape: `(n_samples, n_var)`.
            `out (dict)`: The output dictionary to store the results.
        """

        # Predict the objective metric for the given samples
        top1_err = self.obj_predictor.predict(x)[:, 0]

        # Populate an ouput array with the predicted objective metric
        f = np.full((x.shape[0], self.n_obj), np.nan)
        for i, (_x, err) in enumerate(zip(x, top1_err)):
            encoded_x = _x.astype(np.int8)

            decoded_x = self.search_space.decode(encoded_x)
            subnet = self.engine.get_architecture_model(decoded_x)

            # Get the model size from the net info.
            _x_resolution = decoded_x.get("resolution")
            _x_model_info = get_net_info(subnet, (3, _x_resolution, _x_resolution))

            # Define a metric to use for the objective function
            f[i, 0] = abs(err) * _x_model_info[self.size_metric] ** self.trade_off_param

        # Store the results in the output dictionary
        out["F"] = f

        # Add the non-null direction constraint in case of growing bits
        if isinstance(self.search_space, GrowthSearchSpace):
            out["G"] = 1 - np.sum(x[:, -self.search_space.n_grow_bits :], axis=1)


class MultiObjProblem(Problem):
    def __init__(
        self,
        dataset_name: str,
        ofa_space_family: str,
        search_space: FixedSearchSpace | GrowthSearchSpace,
        first_obj_predictor: SurrogatePredictor,
        second_obj_predictor: SurrogatePredictor,
        trade_off_param: float = 0.07,
        size_metric: str = "params",
    ):
        """
        The optimization problem for finding the next N candidate architectures
        with two objective metrics.

        Note: this is a "single level" problem ( as stated ) in the original paper, so
        the optimisation of both metrics is done at once.
        """
        super().__init__(
            n_var=search_space.nvar,
            n_obj=2,
            n_ieq_constr=1 if isinstance(search_space, GrowthSearchSpace) else 0,
            n_eq_constr=0,
            type_var=np.int8,
        )

        # Define the search space to decode the models
        self.search_space = search_space

        # Define the engine to access the models
        self.engine = get_ofa_evaluator(
            family=ofa_space_family,
            dataset=dataset_name,
        )

        # Define the parameters for the objective function
        self.first_obj_predictor = first_obj_predictor
        self.second_obj_predictor = second_obj_predictor
        self.trade_off_param = trade_off_param
        self.size_metric = size_metric

        # Bounds for the generated X samples (encoded into bit strings)
        self.xl = np.zeros(self.n_var)
        self.xu = 2 * np.ones(self.n_var)

        # Define an upper bound for the resolution
        _res_upper_bound = int(len(self.search_space.input_resolutions) - 1)

        # Check whether we are using expanding architectures
        if isinstance(self.search_space, GrowthSearchSpace):
            self.xu[-1 - self.search_space.n_grow_bits] = _res_upper_bound
        else:
            self.xu[-1] = _res_upper_bound

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        """
        Evaluates the proposed samples using the surrogate model.

        ### Args:
            `x (np.ndarray)`: The proposed candidate architectures. Shape: `(n_samples, n_var)`.
            `out (dict)`: The output dictionary to store the results.
        """

        # Get the first objective metric
        first_obj_metric = self.first_obj_predictor.predict(x)[:, 0]
        second_obj_metric = self.second_obj_predictor.predict(x)[:, 0]

        # Compute the first objective metric
        compound_metric = []
        for err, _x in zip(first_obj_metric, x):
            encoded_x = _x.astype(np.int8)

            # Decode the proposed sample
            decoded_x = self.search_space.decode(encoded_x)
            subnet = self.engine.get_architecture_model(decoded_x)

            # Get the model size from the net info.
            _x_resolution = decoded_x.get("resolution")
            _x_model_info = get_net_info(subnet, (3, _x_resolution, _x_resolution))

            compound_metric.append(
                abs(err) * _x_model_info[self.size_metric] ** self.trade_off_param
            )

        # Populate the output with both objective metric
        f = np.full((x.shape[0], self.n_obj), np.nan)
        for i, (obj1, obj2) in enumerate(zip(compound_metric, second_obj_metric)):
            f[i, 0] = obj1
            f[i, 1] = obj2

        # Store the results in the output dictionary
        out["F"] = f

        # Add the non-null direction constraint in case of growing bits
        if isinstance(self.search_space, GrowthSearchSpace):
            out["G"] = 1 - np.sum(x[:, -self.search_space.n_grow_bits :], axis=1)
