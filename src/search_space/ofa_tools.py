from ofa.utils.pytorch_utils import count_parameters
from torchprofile import profile_macs

import torch.backends.cudnn as cudnn
from copy import deepcopy
from torch import nn
import numpy as np
import torch


target_layers = {
    "Conv2D": nn.Conv2d,
    "Flatten": nn.Flatten,
    "Dense": nn.Linear,
    "BatchNormalization": nn.BatchNorm2d,
    "AveragePooling2D": nn.AvgPool2d,
    "MaxPooling2D": nn.MaxPool2d,
}

activations = {}


def hook_fn(m, i, o):
    # if (o.shape != NULL):
    activations[m] = [i, o]  # .shape  #m is the layer


def get_all_layers(net):
    layers = {}
    names = {}
    index = 0
    for name, layer in net.named_modules():  # net._modules.items():
        # print(name)
        layers[index] = layer
        names[index] = name
        index = index + 1

    # If it is a sequential or a block of modules, don't register a hook on it
    # but recursively register hook on all it's module children
    length = len(layers)
    for i in range(length):
        if i == (length - 1):
            layers[i].register_forward_hook(hook_fn)
        else:
            if (
                (isinstance(layers[i], nn.Sequential))  # sequential
                or (names[i + 1].startswith(names[i] + "."))
            ):  # container of layers
                continue
            else:
                layers[i].register_forward_hook(hook_fn)


def profile_activation_size(model, input):
    activations.clear()
    get_all_layers(model)  # add hooks to model layers
    _out = model(input)  # computes activation while passing through layers

    total = 0

    for name, layer in model.named_modules():
        for label, target in target_layers.items():
            if isinstance(layer, target):
                activation_shape = activations[layer][1].shape
                activation_size = 1
                for i in activation_shape:
                    activation_size = activation_size * i
                total = total + activation_size

    return total


def get_net_info(net, input_shape=(3, 224, 224), print_info=False):
    """
    Modified from https://github.com/mit-han-lab/once-for-all/blob/
    35ddcb9ca30905829480770a6a282d49685aa282/ofa/imagenet_codebase/utils/pytorch_utils.py#L139
    """

    # artificial input data
    inputs = torch.randn(1, 3, input_shape[-2], input_shape[-1])

    # move network to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        net = net.to(device)
        cudnn.benchmark = True
        inputs = inputs.to(device)

    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module

    # this avoids batch norm error https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
    net.eval()

    # Count million of parameters
    n_params = count_parameters(net)
    net_info["params"] = np.round(n_params / 1e6, 2)

    # Count million of MACs
    n_macs = profile_macs(deepcopy(net), inputs)
    net_info["macs"] = np.round(n_macs / 1e6, 2)

    # activation_size
    n_activations = profile_activation_size(net, inputs)
    net_info["activations"] = np.round(n_activations / 1e6, 2)

    if print_info:
        print("Total training params: %.2fM" % (net_info["params"]))
        print("Total MACs: %.2fM" % (net_info["macs"]))
        print("Total activations: %.2fM" % (net_info["activations"]))

    return net_info
