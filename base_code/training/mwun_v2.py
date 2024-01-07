from typing import TypeVar

import torch

from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate

from base_code.helpers import flat_params
from base_code.training.store_loss_base import StoreLossBase


CallbackResult = TypeVar("CallbackResult")


class MWUNPlugin(SupervisedPlugin, StoreLossBase):
    def __init__(
        self,
        lambda_q: float = 1.0,
        lambda_e: float = 1.0,
        lambda_f: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        StoreLossBase.__init__(self)
        self.lambda_q = lambda_q
        self.lambda_e = lambda_e
        self.lambda_f = lambda_f
        self.eps = eps
        self.prev_p: torch.Tensor = None
        self.prev_params: torch.Tensor = None

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        model_params = flat_params(strategy.model)
        self.prev_p = (model_params.abs() > self.eps).float()
        self.prev_params = model_params.clone()

    def before_training(self, strategy: SupervisedTemplate, *args, **kwargs):
        exp_counter = strategy.clock.train_exp_counter

        if exp_counter > 0:
            return

        model_params = flat_params(strategy.model)
        self.prev_p = torch.zeros_like(model_params)
        self.prev_params = model_params.clone()

    def before_backward(
        self, strategy: SupervisedTemplate, *args, **kwargs
    ) -> CallbackResult:
        model_params = flat_params(strategy.model)
        minibatch_length = strategy.mb_x[0].shape[0]
        p = (model_params.abs() > self.eps).float()

        first_component = strategy.loss
        second_component = torch.tensor(0)  # p.norm(1) / p.numel() * minibatch_length
        third_component = (self.prev_p * (model_params - self.prev_params)).norm(1) / (
            (model_params - self.prev_params).norm(2).item() + 1
        )
        fourth_component = ((1 - p) * model_params).norm(1) / ((1 - p).norm(1) + 1)

        # save loss
        self.store_loss(first_component.item(), "first_component")
        self.store_loss(second_component.item(), "second_component")
        self.store_loss(third_component.item(), "third_component")
        self.store_loss(fourth_component.item(), "fourth_component")

        strategy.loss += (
            self.lambda_q * second_component
            + self.lambda_e * third_component
            + self.lambda_f * fourth_component
        )


class MWUN(SupervisedTemplate):
    """Minimum Weight Usage in Networks template.

    MWUN is a training strategy that uses a constrained loss function to
    train a model on a continual learning scenario.

    Relevant parameters:
    --------------------
    lambda_q: float
        The weight of the norm 0 term
    lambda_e: float
        The weight of the elastic term
    lambda_f: float
        The weight of the forced to zero term
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        lambda_q: float = 1.0,
        lambda_e: float = 1.0,
        lambda_f: float = 1.0,
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
        cts = MWUNPlugin(lambda_q, lambda_e, lambda_f, eps)
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

    def get_store_loss(self):
        # search for the CEWCPlugin instance
        for plugin in self.plugins:
            if isinstance(plugin, MWUNPlugin):
                return plugin.get_loss_store()
        return None
