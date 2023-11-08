from typing import Callable, Iterable, Optional, Sequence, Union, Any
from avalanche.core import BasePlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin, default_evaluator
from avalanche.training.templates import SupervisedTemplate
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer


class OGD(SupervisedTemplate):
    """..."""

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=...,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Union[int, None]= 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Union[Sequence[BasePlugin], None] = None,
        evaluator: Union[EvaluationPlugin, Callable[[], EvaluationPlugin]] = ...,
        eval_every=-1,
        peval_mode="epoch"
    ):
        super().__init__(model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device, plugins, evaluator, eval_every, peval_mode)
        self.gradient_basis = []
    
    def backward(self):
        ...
        return super().backward()

    def forward(self):
        return super().forward()
    
    def train(
        self,
        experiences: Union[Any, Iterable],
        eval_streams: Union[Sequence[Any | Iterable], None] = None,
        **kwargs
    ):
        ...
        return super().train(experiences, eval_streams, **kwargs)