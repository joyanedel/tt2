from typing import Any, Dict
from torch import nn
from torch.optim import Optimizer

from base_code.optimizers.base import StatefulOptimizer


class OGD(Optimizer, StatefulOptimizer):
    "Orthogonal Gradient Descent Optimizer algorithm"

    def __init__(self):
        # TODO: take a look to this var (Maybe after it will be required)
        # self.gradient_storage = ...
        self.orthonormal_basis = []

    
    def update(self):
        ...