from ofa.utils.layers import (
    SEModule,
    MBConvLayer,
    ResidualBlock,
    IdentityLayer,
    ConvLayer,
    LinearLayer,
)
from ofa.utils.pytorch_modules import MyGlobalAvgPool2d, Hsigmoid, Hswish
from torch import nn


def transfer_linear_weights(
    source_linear: nn.Linear,
    target_linear: nn.Linear,
    freeze: bool = True,
):
    """
    Copy weights from source to target linear layer.
    Freezes the transfered weights.

    ### Args:
        `source_linear (nn.Linear)`: Source linear layer
        `target_linear (nn.Linear)`: Target linear layer with potentially more features
    """
    out_features, in_features = source_linear.weight.shape

    # Copy existing weights and biases
    target_linear.weight.data[:out_features, :in_features] = source_linear.weight.data
    if source_linear.bias is not None:
        target_linear.bias.data[:out_features] = source_linear.bias.data

    # Freeze the target weights only if there is an exact match
    if freeze and source_linear.weight.shape == target_linear.weight.shape:
        target_linear.requires_grad_(False)


def transfer_conv_weights(
    source_conv: nn.Conv2d,
    target_conv: nn.Conv2d,
    freeze: bool = True,
):
    """
    Copy weights from source to target conv layer, centering the kernel.

    ### Args:
        `source_conv (nn.Conv2d)`: Source convolution layer
        `target_conv (nn.Conv2d)`: Target convolution layer with potentially more channels
        `freeze (bool)`: Whether to freeze the transferred weights
    """
    source_weight = source_conv.weight
    source_out_c, source_in_c, source_k_h, source_k_w = source_weight.shape
    target_out_c, target_in_c, target_k_h, target_k_w = target_conv.weight.shape

    # Determine output channels to transfer
    out_channels = min(source_out_c, target_out_c)
    in_channels = min(source_in_c, target_in_c)

    # Calculate padding for centering when target kernel is larger
    if target_k_h >= source_k_h and target_k_w >= source_k_w:
        pad_h = (target_k_h - source_k_h) // 2
        pad_w = (target_k_w - source_k_w) // 2

        # Copy existing weights to the center
        target_conv.weight.data[
            :out_channels,
            :in_channels,
            pad_h : pad_h + source_k_h,
            pad_w : pad_w + source_k_w,
        ] = source_weight[:out_channels, :in_channels]

        # Freeze the target weights only if there is full overlap
        if freeze and source_conv.weight.shape == target_conv.weight.shape:
            target_conv.requires_grad_(False)

    # Calculate cropping for centering when target is smaller than source
    else:
        crop_h = (source_k_h - target_k_h) // 2
        crop_w = (source_k_w - target_k_w) // 2

        # Copy center weights from source to target
        target_conv.weight.data[:out_channels, :in_channels, :, :] = source_weight[
            :out_channels,
            :in_channels,
            crop_h : crop_h + target_k_h,
            crop_w : crop_w + target_k_w,
        ]


def transfer_bn_weights(
    source_bn: nn.BatchNorm2d,
    target_bn: nn.BatchNorm2d,
    freeze: bool = True,
) -> None:
    """
    Copy weights from source to target batch norm layer.

    ### Args:
        `source_bn (nn.BatchNorm2d)`: Source batch normalization layer
        `target_bn (nn.BatchNorm2d)`: Target batch normalization layer
        `freeze (bool)`: Whether to freeze the transferred weights
    """
    # Get number of features for both BN layers
    source_features = source_bn.num_features
    target_features = target_bn.num_features

    # Only transfer if source has fewer or equal features
    if source_features <= target_features:
        # Copy existing parameters and buffers
        target_bn.weight.data[:source_features] = source_bn.weight.data
        target_bn.bias.data[:source_features] = source_bn.bias.data
        target_bn.running_mean.data[:source_features] = source_bn.running_mean.data
        target_bn.running_var.data[:source_features] = source_bn.running_var.data

        # Freeze the weights in case of exact match
        if freeze and source_features == target_features:
            target_bn.requires_grad_(False)

    # Initialize with same distribution as source
    else:
        target_bn.weight.data.normal_(
            mean=source_bn.weight.data.mean(),
            std=source_bn.weight.data.std(),
        )
        target_bn.bias.data.normal_(
            mean=source_bn.bias.data.mean(),
            std=source_bn.bias.data.std(),
        )
        target_bn.running_mean.data.normal_(
            mean=source_bn.running_mean.data.mean(),
            std=source_bn.running_mean.data.std(),
        )
        target_bn.running_var.data.normal_(
            mean=source_bn.running_var.data.mean(),
            std=source_bn.running_var.data.std(),
        )


def transfer_se_module_weights(
    source_se: SEModule,
    target_se: SEModule,
    freeze: bool = True,
) -> None:
    """
    Copy weights for Squeeze-and-Excitation module.

    Args:
        source_se (SEModule): Source SE module
        target_se (SEModule): Target SE module
    """

    # SE modules typically contain fc layers
    if hasattr(source_se, "fc") and source_se.fc:
        transfer_block_weights(source_se.fc, target_se.fc, freeze)


def transfer_convlayer_weights(
    source_conv: ConvLayer,
    target_conv: ConvLayer,
    freeze: bool = True,
) -> None:
    """
    Copy weights from source to target conv layer.

    ### Args:
        `source_conv (ConvLayer)`: Source convolution layer
        `target_conv (ConvLayer)`: Target convolution layer
    """
    if hasattr(source_conv, "conv") and source_conv.conv:
        transfer_block_weights(source_conv.conv, target_conv.conv, freeze)

    if hasattr(source_conv, "bn") and source_conv.bn:
        transfer_block_weights(source_conv.bn, target_conv.bn, freeze)


def transfer_residual_block_weights(
    source_rb: ResidualBlock,
    target_rb: ResidualBlock,
    freeze: bool = True,
) -> None:
    """
    Copy weights for Residual Block.

    Args:
        source_rb (ResidualBlock): Source Residual Block
        target_rb (ResidualBlock): Target Residual Block
    """
    # Handle main conv path
    if hasattr(source_rb, "conv") and source_rb.conv:
        transfer_block_weights(source_rb.conv, target_rb.conv, freeze)

    # Handle shortcut if not identity
    if hasattr(source_rb, "shortcut") and source_rb.shortcut:
        if not isinstance(source_rb.shortcut, IdentityLayer):
            transfer_block_weights(source_rb.shortcut, target_rb.shortcut, freeze)


def transfer_mb_conv_weights(
    source_mb: MBConvLayer,
    target_mb: MBConvLayer,
    freeze: bool = True,
):
    """Copy weights for Mobile Inverted Bottleneck Conv Layer.

    Args:
        source_mb (MBConvLayer): Source MBConv layer
        target_mb (MBConvLayer): Target MBConv layer
    """
    # Handle inverted bottleneck
    if hasattr(source_mb, "inverted_bottleneck") and source_mb.inverted_bottleneck:
        transfer_block_weights(
            source_mb.inverted_bottleneck,
            target_mb.inverted_bottleneck,
            freeze,
        )

    # Handle depth-wise conv
    if hasattr(source_mb, "depth_conv") and source_mb.depth_conv:
        transfer_block_weights(source_mb.depth_conv, target_mb.depth_conv, freeze)

    # Handle point-wise conv
    if hasattr(source_mb, "point_linear") and source_mb.point_linear:
        transfer_block_weights(source_mb.point_linear, target_mb.point_linear, freeze)

    # Handle SE module if present
    if hasattr(source_mb, "se") and source_mb.se:
        transfer_block_weights(source_mb.se, target_mb.se, freeze)


def transfer_sequential_weights(
    source_seq: nn.Sequential,
    target_seq: nn.Sequential,
    freeze: bool = True,
):
    """
    Copy weights for Sequential container.

    ### Args:
        `source_seq (nn.Sequential)`: Source Sequential container
        `target_seq (nn.Sequential)`: Target Sequential container
    """
    for source_block, target_block in zip(source_seq.children(), target_seq.children()):
        transfer_block_weights(
            source_block,
            target_block,
            freeze,
        )


def transfer_block_weights(
    source_block: nn.Module,
    target_block: nn.Module,
    freeze: bool = True,
) -> None:
    """
    Transfer weights from source to target block, handling the different block types.

    ### Args:
        `source_block (nn.Module)`: Source block from base network
        `target_block (nn.Module)`: Target block from expanded network
    """

    if isinstance(source_block, nn.Sequential):
        transfer_sequential_weights(source_block, target_block, freeze)

    elif isinstance(source_block, nn.Conv2d):
        transfer_conv_weights(source_block, target_block, freeze)

    elif isinstance(source_block, nn.BatchNorm2d):
        transfer_bn_weights(source_block, target_block, freeze)

    elif isinstance(source_block, nn.Linear):
        transfer_linear_weights(source_block, target_block, freeze)

    elif isinstance(source_block, LinearLayer):
        transfer_linear_weights(source_block.linear, target_block.linear, freeze)

    elif isinstance(source_block, SEModule):
        transfer_se_module_weights(source_block, target_block, freeze)

    elif isinstance(source_block, ConvLayer):
        transfer_convlayer_weights(source_block, target_block, freeze)

    elif isinstance(source_block, ResidualBlock):
        transfer_residual_block_weights(source_block, target_block, freeze)

    elif isinstance(source_block, MBConvLayer):
        transfer_mb_conv_weights(source_block, target_block, freeze)

    elif isinstance(
        source_block,
        (IdentityLayer, nn.Identity, nn.ReLU, Hsigmoid, Hswish, MyGlobalAvgPool2d),
    ):
        # Activation layers don't have weights to transfer
        pass

    else:
        print(f"Warning: Unhandled layer type: {type(source_block)}")
