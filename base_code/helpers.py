import numpy as np
from typing import Dict


import torch
from torch.nn import Module
from avalanche.training.utils import ParamData


def to_categorical(y: int, num_classes: int):
    """Creates a 1-hot encoded array from a given integer
    Parameters
    ----------
    y: int
        Integer to be converted to 1-hot encoded array
    num_classes: int
        Number of classes in the dataset

    Returns
    -------
    np.ndarray
        1-hot encoded array
    """

    return np.eye(num_classes)[y]


def flat_grads(model: Module) -> torch.Tensor:
    grads = [param.grad.view(-1) for _, param in model.named_parameters()]

    return torch.cat(grads)


def flat_params(model: Module) -> torch.Tensor:
    weights = [param.view(-1) for _, param in model.named_parameters()]

    return torch.cat(weights)

def flat_importances(importances: Dict[str, ParamData]) -> torch.Tensor:
    importances = [importance._data.view(-1) for _, importance in importances.items()]

    return torch.cat(importances)