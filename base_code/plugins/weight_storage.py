from typing import List

import torch

from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate

from base_code.helpers import flat_params


class WeightStoragePlugin(SupervisedPlugin):
    """Weight Storage Plugin.

    This plugin stores the model weights at the end of each experience.
    """

    def __init__(self):
        self.weights: List[torch.Tensor] = []

    def before_training(self, strategy: SupervisedTemplate, *args, **kwargs):
        self.weights.append(flat_params(strategy.model).detach().clone().cpu())

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        self.weights.append(flat_params(strategy.model).detach().clone().cpu())
