from search_space._weights_transfers import transfer_block_weights
from search_space.growth_space import GrowthSearchSpace
from search_space.base_ofa import OFAEvaluator
from search_space.base_space import ModelSample

from copy import deepcopy
from torch import nn

from typing import List


def _transfer_base_structure(
    base_model: nn.Module,
    ofa_model: nn.Module,
    freeze_base: bool = True,
) -> None:
    """
    Transfer the models structures not expanded by the OFA.
    """
    # Transfer the rest of the model structures
    transfer_block_weights(
        base_model.first_conv,
        ofa_model.first_conv,
        freeze_base,
    )
    transfer_block_weights(
        base_model.final_expand_layer,
        ofa_model.final_expand_layer,
        freeze_base,
    )
    transfer_block_weights(
        base_model.feature_mix_layer,
        ofa_model.feature_mix_layer,
        freeze_base,
    )
    transfer_block_weights(
        base_model.classifier,
        ofa_model.classifier,
        freeze_base,
    )


def _expand_shallow_dimension(
    base_model: nn.Module,
    model_sample: ModelSample,
    search_space: GrowthSearchSpace,
    ofa_evaluator: OFAEvaluator,
    scaling_vector: List[int],
    freeze_base: bool = True,
) -> tuple[ModelSample, nn.Module]:
    """
    Expands a model along a specific SHALLOW dimension (width or kernel size).

    Args:
        base_model: Current model to expand
        model_sample: Current model architecture definition
        search_space: Search space containing valid architectures
        ofa_evaluator: Evaluator to obtain OFA weights
        scaling_vector: Vector indicating scaling direction [depth, width, kernel]

    Returns:
        tuple[ModelSample, nn.Module]: Updated model definition and expanded model
    """

    # Obtain the expanded architecture sample
    expanded_sample = search_space.apply_scaling(model_sample, scaling_vector)

    # Obtain the OFA weights for the expanded architecture
    ofa_model = ofa_evaluator.get_architecture_model(expanded_sample)
    ofa_model = ofa_model.cpu()

    # Transfer the base model structures
    _transfer_base_structure(base_model, ofa_model, freeze_base)

    # Transfer the weights from the base model to the expanded architecture
    for bblock, ofablock in zip(base_model.blocks, ofa_model.blocks):
        transfer_block_weights(bblock, ofablock, freeze_base)

    # Return to update model and sample
    return expanded_sample, ofa_model


def _expand_depth_dimension(
    base_model: nn.Module,
    model_sample: ModelSample,
    search_space: GrowthSearchSpace,
    ofa_evaluator: OFAEvaluator,
    scaling_vector: List[int],
    freeze_base: bool = True,
) -> tuple[ModelSample, nn.Module]:
    """
    Expands a model along a specific DEPTH dimension. After expanding
    it matches the number of blocks on the model so the transfer of the
    weights can transfer as many as possible values.

    Args:
        base_model: Current model to expand
        model_sample: Current model architecture definition
        search_space: Search space containing valid architectures
        ofa_evaluator: Evaluator to obtain OFA weights
        scaling_vector: Vector indicating scaling direction [depth, width, kernel]

    Returns:
        tuple[ModelSample, nn.Module]: Updated model definition and expanded model
    """
    # Obtain the expanded architecture with the depth
    expanded_sample = search_space.apply_scaling(model_sample, scaling_vector)

    # Obtain the OFA weights for the expanded architecture
    ofa_model = ofa_evaluator.get_architecture_model(expanded_sample)
    ofa_model = ofa_model.cpu()

    # Match the depth of the expanded model by copying the last base model block
    base_depths = model_sample["depths"]
    expanded_depths = expanded_sample["depths"]

    if base_depths != expanded_depths:
        # Find where the depth increased
        added_block_idx = None
        for i, (base_d, exp_d) in enumerate(zip(base_depths, expanded_depths)):
            if exp_d > base_d:
                added_block_idx = i
                break

        # Get the last common block among the models
        last_common_block = (
            1 + sum(base_depths[:added_block_idx]) + base_depths[added_block_idx]
        )

        # Insert the last common block of the base model into the expanded model
        base_blocks = deepcopy(base_model.blocks)
        base_blocks.insert(
            last_common_block,
            base_blocks[last_common_block - 1],
        )
    else:
        base_blocks = deepcopy(base_model.blocks)

    # Transfer the weights from the base model to the expanded architecture
    for i, (bblock, ofablock) in enumerate(zip(base_blocks, ofa_model.blocks)):
        transfer_block_weights(bblock, ofablock, freeze_base)

    # Transfer the rest of the model structures
    _transfer_base_structure(base_model, ofa_model, freeze_base)

    return expanded_sample, ofa_model


def apply_model_expansion(
    base_model: nn.Module,
    model_sample: ModelSample,
    search_space: GrowthSearchSpace,
    ofa_evaluator: OFAEvaluator,
    scaling_direction: List[int],
    freeze_base: bool = True,
) -> tuple[ModelSample, nn.Module]:
    """
    Expands the model, both in the SearchSpace definition space and in the weights space.
    To retain the most information from the base model, the expansions are performed in the
    following order:

        1) Kernel sizes
        2) Widths
        3) Depths

    ### Returns
        `ModelSample`: the updated model sample definition
        `nn.Module`: the expanded model
    """
    ddir, wdir, kdir = scaling_direction

    # Move base model to cpu for memory efficiency
    expanded_model = base_model.cpu()

    # First expands the kernel sizes of the base model
    if kdir > 0:
        model_sample, expanded_model = _expand_shallow_dimension(
            expanded_model,
            model_sample,
            search_space,
            ofa_evaluator,
            scaling_vector=[0, 0, kdir],
            freeze_base=freeze_base,
        )

    # Expand the filters of the base model
    if wdir > 0:
        model_sample, expanded_model = _expand_shallow_dimension(
            expanded_model,
            model_sample,
            search_space,
            ofa_evaluator,
            scaling_vector=[0, wdir, 0],
            freeze_base=freeze_base,
        )

    if ddir > 0:
        model_sample, expanded_model = _expand_depth_dimension(
            expanded_model,
            model_sample,
            search_space,
            ofa_evaluator,
            scaling_vector=[ddir, 0, 0],
            freeze_base=freeze_base,
        )

    # Add the direction to the model sample
    model_sample["direction"] = scaling_direction

    return model_sample, expanded_model
