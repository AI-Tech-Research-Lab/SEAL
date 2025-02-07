from tools.metrics import binary_accuracy

from torch.utils.data import DataLoader, Subset
from copy import deepcopy
import numpy as np
import torch

from typing import List


def validate_accuracy(
    model: torch.nn.Module,
    task_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Perform accuracy validation on a task.

    ### Args:
        `model (nn.Module)`: The model to validate.
        `task_loader (DataLoader)`: DataLoader for validation data.
        `device (torch.device)`: Device to use for validation.

    ### Returns:
        `float`: Average accuracy over the validation set.
    """
    model.eval()
    accuracies = []

    # Iterate over the dataset without gradients tracking
    with torch.no_grad():
        eval_model = model.to(device)
        for inputs, labels in task_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Compute accuracy
            outputs = eval_model(inputs)
            acc = binary_accuracy(outputs, labels).item()

            # Explicitly delete tensors after each batch
            del inputs, labels, outputs
            accuracies.append(acc)

        # Clean up unused gpu memory
        torch.cuda.empty_cache()

    return np.array(accuracies).mean()


def create_model_noises(
    model: torch.nn.Module,
    noise: str = "mult",
) -> List[torch.Tensor]:
    """
    Uses the model parameters to create a perturbation vector for each weight in the
    model.

    Additive noise makes the noise weight to be a scaled vector of the initial weights.

    ### Args:
        `model (torch.nn.Module)`: The model to perturb.
        `noise (str)`: The type of noise to add. Can be "mult" for multiplicative noise
            or "add" for additive noise.

    ### Returns:
        `List[torch.Tensor]`: A list of perturbation vectors.

    """
    z = []
    for p in model.parameters():
        # Create a normal noise with the shape of the weights
        r = p.clone().detach().normal_()

        # Multiplicative noise
        if noise == "mult":
            z.append(p.data * r)

        # Additive noise
        else:
            z.append(r)

    return z


def perturb_model(
    model: torch.nn.Module,
    z: List[torch.Tensor],
    noise_ampl: float,
):
    """
    Perturbs a model by summing the noise to the original model weights.

    ### Args:
        `model (torch.nn.Module)`: The model to perturb.
        `z (List[torch.Tensor])`: The perturbation vectors.
        `noise_ampl (float)`: The scaling of the noise to add.

    ### Returns:
        `torch.nn.Module`: The perturbed model.
    """
    for i, p in enumerate(model.parameters()):
        p.data += noise_ampl * z[i]


def calculate_mean_flatness(
    model: torch.nn.Module,
    task_loader: DataLoader,
    sigma: float,
    device: torch.device,
    subsample_perc: float = 0.1,
    neighbourhood_size: int = 20,
) -> float:
    """
    Computes the mean flatness of a model weights over a "random" neighbourhood of the
    model.
    """

    # Create a base model for the perturbations
    perturbed_model = deepcopy(model)

    # Get initial accuracy, without any perturbation
    acc_pre = validate_accuracy(perturbed_model, task_loader, device)

    flatness_list = []
    with torch.no_grad():
        for _ in range(neighbourhood_size):
            # Perturb model
            model_noises = create_model_noises(perturbed_model)
            perturb_model(perturbed_model, model_noises, sigma)

            # Create a subsample of the loader to compute the flatness
            subset_size = int(len(task_loader.dataset) * subsample_perc)
            sample_indices = torch.randperm(len(task_loader.dataset))[:subset_size]
            subset_loader = DataLoader(
                dataset=Subset(task_loader.dataset, sample_indices),
                batch_size=task_loader.batch_size,
                num_workers=task_loader.num_workers,
                pin_memory=False,
                shuffle=False,
            )

            # Compute the perturbed model's accuracy
            perturbed_acc = validate_accuracy(perturbed_model, subset_loader, device)

            # Compute the delta in accuracy
            acc_delta = abs(acc_pre - perturbed_acc)
            flatness = (acc_pre - acc_delta) / acc_pre
            flatness_list.append(flatness)

            # Unperturb the model to go back to the initial state
            # This is a trick to avoid OOM issues when on small GPUs.
            # It may introduce small numerical instability.
            perturb_model(perturbed_model, model_noises, -sigma)

            # Clean up iteration variables
            torch.cuda.empty_cache()

    # Cleanup
    del model_noises
    del perturbed_model
    torch.cuda.empty_cache()

    return sum(flatness_list) / len(flatness_list)
