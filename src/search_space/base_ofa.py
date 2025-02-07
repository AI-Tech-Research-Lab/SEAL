from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicSeparableConv2d,
)
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
import warnings
import torch

from typing import Dict

from _constants import OFA_MODEL_PATH

warnings.simplefilter("ignore")

DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1


class OFAEvaluator:
    def __init__(
        self,
        family: str = "mobilenetv3",
        model_path: str = "./ofa_nets/ofa_mbv3_d234_e346_k357_w1.0",
        data_classes: int = 1000,
        pretrained: bool = False,
    ):
        """
        Initialize the OFAEvaluator.

        ### Args:
            `family (str)`: The OFA family type. Currently only supports "mobilenetv3".
            `model_path (str)`: Path to the pretrained model weights.
            `data_classes (int)`: Number of output classes.
            `pretrained (bool)`: Whether to load pretrained weights.

        ### Raises:
            `KeyError`: If an unsupported OFA family type is provided.
            `ValueError`: If an invalid model path is provided.
        """
        match family:
            case "mobilenetv3":
                self.engine = self._create_engine(
                    "mobilenetv3", OFA_MODEL_PATH[family], data_classes
                )
            case _:
                raise KeyError(f"OFA family type: '{family}' not implemented!")

        if pretrained:
            self._load_pretrained_weights(model_path, data_classes)

    def _create_engine(
        self,
        family: str,
        model_path: str,
        data_classes: int,
    ) -> OFAMobileNetV3:
        """
        Create the OFA engine based on the family type.

        Args:
            family (str): The OFA family type.
            model_path (str): Path to the pretrained model weights.
            data_classes (int): Number of output classes.

        Returns:
            OFAMobileNetV3: The created OFA engine.

        Raises:
            KeyError: If an unsupported OFA family type is provided.
            ValueError: If an invalid model path is provided.
        """
        match family:
            case "mobilenetv3":
                width_mult = self._get_width_mult(model_path)
                return OFAMobileNetV3(
                    n_classes=data_classes,
                    dropout_rate=0,
                    width_mult=width_mult,
                    depth_list=[2, 3, 4],
                    ks_list=[3, 5, 7],
                    expand_ratio_list=[3, 4, 6],
                )

            case _:
                raise KeyError(f"OFA family type: '{family}' not implemented!")

    @staticmethod
    def _get_width_mult(model_path: str) -> float:
        """
        Get the width multiplier from the model path.

        Args:
            model_path (str): Path to the pretrained model weights.

        Returns:
            float: The width multiplier.

        Raises:
            ValueError: If an invalid model path is provided.
        """
        if "w1.0" in model_path or "resnet50" in model_path:
            return 1.0
        elif "w1.2" in model_path:
            return 1.2
        else:
            raise ValueError("Invalid model path: unable to determine width multiplier")

    def _load_pretrained_weights(self, model_path: str, data_classes: int) -> None:
        """
        Load pretrained weights and fix size mismatch if necessary.

        Args:
            model_path (str): Path to the pretrained model weights.
            data_classes (int): Number of output classes.
        """
        init = torch.load(model_path, map_location="cpu")["state_dict"]

        # Fix size mismatch error
        init["classifier.linear.weight"] = init["classifier.linear.weight"][
            :data_classes
        ]
        init["classifier.linear.bias"] = init["classifier.linear.bias"][:data_classes]

        self.engine.load_state_dict(init)

    def get_architecture_model(
        self,
        architecture: Dict[str, list],
        pretrained: bool = True,
    ) -> torch.nn.Module:
        """
        Get a subnet model based on the provided architecture or sample a random one.

        ### Args:
            `architecture` (Dict[str, list]): A dictionary containing the architecture
                specification with keys 'depths', 'widths', and 'ksizes'. If None, a random
                architecture will be sampled.

        Returns:
            torch.nn.Module: The subnet model.

        Raises:
            `ValueError`: If an invalid architecture is provided.
        """
        try:
            self.engine.set_active_subnet(
                d=architecture["depths"],
                e=architecture["widths"],
                ks=architecture["ksizes"],
            )
            subnet = self.engine.get_active_subnet(preserve_weight=pretrained)
            return subnet
        except KeyError:
            raise ValueError("Invalid architecture provided")
