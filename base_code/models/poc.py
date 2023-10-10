import torch

from torch import nn
from torch import Tensor


class NeuralNetwork(nn.Module):
    def __init__(self, features_shape: int, output_shape) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(features_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_shape),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor):
        x = x.type(torch.float32)
        print(x)
        logits = self.linear_relu_stack(x)
        print(logits)
        return logits
