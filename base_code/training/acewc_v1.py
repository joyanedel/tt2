from typing import TypeVar, List

import torch
from torch.utils.data import DataLoader
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.utils import zerolike_params_dict

from base_code.helpers import flat_params, flat_importances

CallbackResult = TypeVar("CallbackResult")


class ACEWCPlugin(SupervisedPlugin):
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
        self.used_params: List[torch.Tensor] = list()
        self.saved_params: List[torch.Tensor] = list()


    def before_backward(self, strategy: SupervisedTemplate, *args, **kwargs) -> CallbackResult:
        exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(strategy.device)

        flatten_params = flat_params(strategy.model)

        for prev_exp in range(exp_counter):
            used_params = self.used_params[prev_exp]
            not_used_params = ~used_params
            diff_weights = flatten_params - self.saved_params[prev_exp]

            penalty += (diff_weights * used_params).norm(1) * self.ewc_lambda / exp_counter
            penalty += (diff_weights * not_used_params).norm(2) * self.ewc_lambda / exp_counter

        strategy.loss += penalty

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        importances = self.compute_importances(
            strategy.model,
            strategy._criterion,
            strategy.optimizer,
            strategy.experience.dataset,
            strategy.device,
            strategy.train_mb_size,
        )

        new_used = importances > self.threshold
        
        self.used_params.append(new_used)
        self.saved_params.append(flat_params(strategy.model).clone())

    def compute_importances(
        self, model: torch.nn.Module, criterion, optimizer, dataset, device, batch_size, num_workers=0
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

            for (k1, p), (k2, imp) in zip(model.named_parameters(), importances.items()):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        model.train()

        return flat_importances(importances)


class ACEWC(SupervisedTemplate):
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
        **base_kwargs,
    ):
        cewc = ACEWCPlugin(ewc_lambda, threshold)

        if plugins is None:
            plugins = []
        plugins.append(cewc)

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
