from .base_models import get_base_predictor, BaseSurrogatePredictor
from .adaptative_switching import AdaptiveSwitching
from _utils import set_seed

import numpy as np

from typing import TypeAlias

SurrogatePredictor: TypeAlias = AdaptiveSwitching | BaseSurrogatePredictor


def get_surrogate_predictor(
    model_name: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    random_seed: int = 42,
) -> SurrogatePredictor:
    """
    Wrapps the interface with all the surrogate models.

    ### Args
        `model_name (str)`: Name of the surrogate model to use.
        `inputs (np.ndarray)`: Input data for training the surrogate model.
        `targets (np.ndarray)`: Target values for training the surrogate model.

    ### Returns
        `SurrogatePredictor`: Trained surrogate predictor.
    """
    set_seed(random_seed)

    # In case we are not interested in using adaptive switching
    if model_name not in ["as", "adaptive switching"]:
        predictor = get_base_predictor(model_name, inputs, targets)
        return predictor

    # In case we are using adaptive switching
    predictor = AdaptiveSwitching()
    predictor.fit(inputs, targets)
    return predictor
