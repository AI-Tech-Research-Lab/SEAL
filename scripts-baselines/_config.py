import warnings
import sys
import os


from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicSeparableConv2d,
)


warnings.simplefilter("ignore")

# Add the src/ dir to the path of import directory
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
    )
)

# Define the OFA family
OFA_FAMILY = "mobilenetv3"

DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1
