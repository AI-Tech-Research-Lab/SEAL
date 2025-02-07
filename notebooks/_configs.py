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

DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1

OFA_MODEL_PATH = "assets/supernets/ofa_mbv3_d234_e346_k357_w1.0"
