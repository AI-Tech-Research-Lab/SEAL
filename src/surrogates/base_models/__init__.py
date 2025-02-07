import numpy as np

from .neighbors import Neighbors
from .random_forest import RandomForestModel
from .svr import SVRModel
from .carts import CART
from .mlp import MLP
from .gpr import GPR


from typing import TypeAlias

BaseSurrogatePredictor: TypeAlias = (
    Neighbors | CART | MLP | GPR | RandomForestModel | SVRModel
)


def get_base_predictor(
    model_name: str,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> BaseSurrogatePredictor:
    """
    Defines and trains individual base surrogate predictors based on the given
    inputs and targets.

    ### Args
        `model_name (str)`: Name of the surrogate model to use.
        `inputs (np.ndarray)`: Input data for training the surrogate model.
        `targets (np.ndarray)`: Target values for training the surrogate model.

    ### Returns
        `SurrogatePredictor`: Trained surrogate predictor.
    """
    test_msg = "# of training samples have to be > # of dimensions"
    assert len(inputs) > len(inputs[0]), test_msg

    if model_name == "neighbors":
        acc_predictor = Neighbors()
        acc_predictor.fit(inputs, targets)

    elif model_name == "carts":
        acc_predictor = CART(n_tree=5000)
        acc_predictor.fit(inputs, targets)

    elif model_name == "gpr":
        acc_predictor = GPR()
        acc_predictor.fit(inputs, targets)

    elif model_name == "mlp":
        acc_predictor = MLP(input_dim=inputs.shape[1])
        acc_predictor.fit(x=inputs, y=targets)

    elif model_name == "random_forest":
        acc_predictor = RandomForestModel()
        acc_predictor.fit(inputs, targets)

    elif model_name == "svr":
        acc_predictor = SVRModel()
        acc_predictor.fit(inputs, targets)

    else:
        raise NotImplementedError

    return acc_predictor
