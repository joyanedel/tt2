from typing import TypeVar, List, Dict, Union
from avalanche.benchmarks import CLExperience, CLStream
from contextlib import contextmanager
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.utils import zerolike_params_dict

from base_code.helpers import flat_params, flat_importances

CallbackResult = TypeVar("CallbackResult")


class EnsembleHardVotingModel(torch.nn.Module):
    def __init__(self, models: List[torch.nn.Module]):
        super().__init__()
        self.models = models

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mode(dim=0).values


class EEWCPlugin(SupervisedPlugin):
    """
    Constrained Elastic Weight Consolidation (CEWC) plugin.
    """

    def __init__(
        self,
        ewc_lambda: float,
        threshold: float = 1.0,
    ):
        self.ewc_lambda = ewc_lambda
        self.threshold = threshold
        self.importances: torch.Tensor = None
        """Dict of importances for each experience"""
        self.saved_params: Dict[int, any] = dict()
        """Dict of saved parameters for each experience"""
        self.prev_params: torch.Tensor = None

    def before_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        if self.prev_params is None:
            self.prev_params = torch.zeros_like(flat_params(strategy.model)).to(
                strategy.device
            )

        if self.importances is None:
            self.importances = torch.zeros_like(flat_params(strategy.model)).to(
                strategy.device
            )

    def before_backward(
        self, strategy: SupervisedTemplate, *args, **kwargs
    ) -> CallbackResult:
        current_flatten_params = flat_params(strategy.model)

        diff_weights = current_flatten_params - self.prev_params

        strategy.loss += (
            self.ewc_lambda * (diff_weights * self.importances).norm(2) ** 2
        )

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        importances = self.compute_importances(
            strategy.model,
            strategy._criterion,
            strategy.optimizer,
            strategy.experience.dataset,
            strategy.device,
            strategy.train_mb_size,
        )
        self.importances = importances
        self.saved_params[strategy.clock.train_exp_counter] = deepcopy(
            strategy.model.state_dict()
        )
        self.prev_params = flat_params(strategy.model)

    def compute_importances(
        self,
        model: torch.nn.Module,
        criterion,
        optimizer,
        dataset,
        device,
        batch_size,
        num_workers=0,
    ) -> torch.Tensor:
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)

        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        for batch in dataloader:
            # get only input, target and task_id from the batch
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                model.named_parameters(), importances.items()
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        model.train()

        return flat_importances(importances)

    def get_all_stored_weights(self):
        return self.saved_params


class EEWC(SupervisedTemplate):
    """Acumulative Constrains over Elastic Weight Consolidation (CEWC) strategy implementation"""

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        ewc_lambda: float = 1.0,
        threshold: float = 1.0,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device="cpu",
        plugins=None,
        evaluator=None,
        eval_every=-1,
        model_arch: dict = None,
        **base_kwargs,
    ):
        cewc = EEWCPlugin(ewc_lambda, threshold)

        if plugins is None:
            plugins = []
        plugins.append(cewc)
        self.model_arch = model_arch

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

    def get_all_stored_weights(self):
        for plugin in self.plugins:
            if isinstance(plugin, EEWCPlugin):
                return plugin.get_all_stored_weights()
        return None

    @contextmanager
    def ensemble_model(self):
        # construct an ensemble of models where each model used already store weights from plugin
        # and evaluate this ensemble

        # save original model
        original_model = self.model

        # get all stored weights
        try:
            stored_weights = self.get_all_stored_weights()

            # create a list of models
            models = []
            for _, weights in stored_weights.items():
                model = self.model.__class__(**self.model_arch)
                model.load_state_dict(weights)
                models.append(model)

            # evaluate per model
            self.model = EnsembleHardVotingModel(models)

            yield

        finally:
            self.model = original_model

    def eval(self, exp_list: Union[CLExperience, CLStream], **kwargs):
        # construct an ensemble of models where each model used already store weights from plugin
        # and evaluate this ensemble

        with self.ensemble_model() as _:
            return super().eval(exp_list, **kwargs)
