# ruff: noqa: F401
from _constants import DATASET_N_CLASSES, OFA_MODEL_PATH

from search_space.base_space import BaseOFASearchSpace, ModelSample
from search_space.base_ofa import OFAEvaluator
from .fixed_space import FixedSearchSpace
from .growth_space import GrowthSearchSpace


def get_search_space(family: str, fixed: bool = False) -> BaseOFASearchSpace:
    # Get the correct data class for the fixed or growing case
    space_class = FixedSearchSpace if fixed else GrowthSearchSpace

    match family:
        case "mobilenetv3":
            return space_class(family="mobilenetv3")

        case _:
            raise ValueError(f"Unknown search space family: {family}")


def get_ofa_evaluator(
    family: str,
    dataset: str,
    pretrained: bool = True,
) -> OFAEvaluator:
    match family:
        case "mobilenetv3":
            return OFAEvaluator(
                family=family,
                model_path=OFA_MODEL_PATH[family],
                data_classes=DATASET_N_CLASSES[dataset],
                pretrained=pretrained,
            )

        case _:
            raise ValueError(f"Unknown search space family: {family}")
