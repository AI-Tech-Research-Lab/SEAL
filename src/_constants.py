from custom.imagenet16 import ImageNet16
import torchvision.datasets as datasets

BASE_DATASET_DIR = "assets/datasets"

OFA_MODEL_PATH = {
    "mobilenetv3": "assets/supernets/ofa_mbv3_d234_e346_k357_w1.0",
    "mobilenetv3-growth": "assets/supernets/ofa_mbv3_d234_e346_k357_w1.0",
}

DATASET_MEAN_STD = {
    "cifar10": {
        "mean": [0.49139968, 0.48215827, 0.44653124],
        "std": [0.24703233, 0.24348505, 0.26158768],
    },
    "cifar100": {
        "mean": [0.50707516, 0.48654887, 0.44091784],
        "std": [0.26733429, 0.25643846, 0.27615047],
    },
    "imagenet16": {
        "mean": [x / 255 for x in [122.68, 116.66, 104.01]],
        "std": [x / 255 for x in [63.22, 61.26, 65.09]],
    },
    "imagenet": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}

DATASET_MAPPING = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "imagenet": datasets.ImageNet,
    "imagenet16": ImageNet16,
}

DATASET_N_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet16": 120,
    "imagenet": 1000,
}

SINGLE_OBJ_TRADE_OFF_PARAM = 0.07
