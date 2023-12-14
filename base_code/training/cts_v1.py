from typing import TypeVar

import torch

from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate

from base_code.helpers import flat_params


CallbackResult = TypeVar("CallbackResult")


class CTSPlugin(SupervisedPlugin):
    def __init__(
        self, lambda_l1: float = 1.0, lambda_l2: float = 1.0, eps: float = 1e-8
    ):
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.eps = eps

    def before_backward(
        self, strategy: SupervisedTemplate, *args, **kwargs
    ) -> CallbackResult:
        exp_counter = strategy.clock.train_exp_counter

        model_params = flat_params(strategy.model)
        l2 = self.lambda_l2 if exp_counter > 0 else 0
        s = self.get_s(strategy.model, exp_counter == 0)
        divider_1 = ((s > 0.5).sum().item() + 1) ** 0.5
        divider_2 = ((s <= 0.5).sum().item() + 1) ** 0.5

        strategy.loss += (self.lambda_l1 / divider_1) * (s * model_params).norm(1)
        strategy.loss += (l2 / divider_2) * ((1 - s) * model_params).norm(2) ** 2

    def get_s(self, model: torch.nn.Module, first_iter: bool) -> torch.Tensor:
        """Returns a binary mask that indicates which parameters are
        constrained and which are not.

        The constrained parameters are those that have a small absolute value, which means that are not being used by the model still.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.
        first_iter : bool
            Whether this is the first iteration of the training loop.

        Returns
        -------
        torch.Tensor
            A binary mask indicating which parameters are constrained.
        """
        if first_iter:
            return torch.ones_like(flat_params(model))

        flatten_params = flat_params(model)

        return (flatten_params.abs() <= self.eps).float()


class CTS(SupervisedTemplate):
    """Constrained Training Strategy (CTS) template.

    CTS is a training strategy that uses a constrained loss function to
    train a model on a continual learning scenario.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        lambda_l1: float = 1.0,
        lambda_l2: float = 1.0,
        eps: float = 1e-8,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device="cpu",
        plugins=None,
        evaluator=None,
        eval_every=-1,
        **base_kwargs,
    ):
        cts = CTSPlugin(lambda_l1, lambda_l2, eps)

        if plugins is None:
            plugins = []
        plugins.append(cts)

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs,
        )
