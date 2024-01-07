from copy import deepcopy
from typing import Callable, List, Dict, Sequence, Union, Set
from contextlib import contextmanager
from avalanche.benchmarks import CLExperience, CLStream

import torch
from avalanche.core import BasePlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from torch._C import device
from torch.nn import Module
from torch.optim import Optimizer


class EnsembleHardVotingStrategy(SupervisedTemplate):
    """Strategy that performs hard voting on a list of models."""

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=...,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Union[int, None] = 1,
        device: Union[str, device] = "cpu",
        plugins: Union[Sequence[BasePlugin], None] = None,
        evaluator: Union[EvaluationPlugin, Callable[[], EvaluationPlugin]] = ...,
        eval_every=-1,
        peval_mode="epoch",
    ):
        store_plugin = StoreModelParamsPlugin()
        if plugins is None:
            plugins = []

        plugins.append(store_plugin)

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )

    def get_all_stored_weights(self):
        """Returns a dictionary of all stored weights."""
        for plugin in self.plugins:
            if isinstance(plugin, StoreModelParamsPlugin):
                return plugin.saved_params
        return None

    def get_observed_classes_per_exp(self):
        """Returns a dictionary of all observed classes per experience."""
        for plugin in self.plugins:
            if isinstance(plugin, StoreModelParamsPlugin):
                return plugin.observed_classes_per_exp
        return None

    @contextmanager
    def ensemble_model(self):
        """Context manager that yields an ensemble of models."""
        # construct an ensemble of models where each model used already store weights from plugin
        # and evaluate this ensemble
        yield
        # training_model = deepcopy(self.model)

        # try:
        #     saved_params = self.get_all_stored_weights()
        #     models = []
        #     for params in saved_params.values():
        #         model = deepcopy(self.model)
        #         model.load_state_dict(params)
        #         models.append(model)
        #     # ensemble_model = EnsembleHardVotingModel(models)
        #     self.model = training_model
        #     yield

        # finally:
        #     self.model = training_model

    def eval(self, exp_list: Union[CLExperience, CLStream], **kwargs):
        """Evaluates the strategy."""
        training_model = deepcopy(self.model)

        # check if model parameters are the same as the ones stored in the plugin
        # if not, raise an error

        saved_params = self.get_all_stored_weights()
        observed_classes = self.get_observed_classes_per_exp()

        try:
            models = []
            observed_classes_per_exp = []
            for exp, params in saved_params.items():
                model = deepcopy(self.model)
                model.load_state_dict(params, assign=True)
                model.eval()
                models.append(model)
                observed_classes_per_exp.append(observed_classes[exp])
            ensemble_model = EnsembleHardVotingModel(
                models, observed_classes, observed_classes_per_exp
            )
            ensemble_model.eval()
            self.model = ensemble_model
            self.model.eval()
            metrics = super().eval(exp_list, **kwargs)
        finally:
            self.model = training_model
            self.model.train()

        return metrics


class StoreModelParamsPlugin(SupervisedPlugin):
    """Plugin that stores the parameters of the model after each experience."""

    def __init__(self):
        self.saved_params: Dict[int, any] = dict()
        self.observed_classes_per_exp: Dict[int, List[int]] = dict()
        self.observed_classes_so_far: Set[int] = set()

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        """Stores the parameters of the model after each experience."""
        self.observed_classes_so_far.update(
            strategy.experience.classes_in_this_experience
        )

        self.saved_params[strategy.clock.train_exp_counter] = deepcopy(
            strategy.model.state_dict(keep_vars=False)
        )
        self.observed_classes_per_exp[strategy.clock.train_exp_counter] = deepcopy(
            list(self.observed_classes_so_far)
        )


class EnsembleHardVotingModel(torch.nn.Module):
    """Embedding model that performs hard voting on a list of models."""

    def __init__(
        self,
        models: List[torch.nn.Module],
        observed_classes: Dict[int, List[int]],
        observed_classes_per_exp: List[List[int]],
    ):
        super().__init__()
        self.models = models
        self.observed_classes = observed_classes
        self.observed_classes_per_exp = observed_classes_per_exp

    def forward(self, x):
        outputs = [model(x) for model in self.models]

        # in order to perform hard voting, we need to know what classes a model has seen, if a model has not seen a class
        # and the model predicts this class, we need to ignore this prediction
        # but if a model has seen a class and predicts this class, we need to keep this prediction.
        # Also if a model predicts a class that no model has seen, we need to keep this prediction to avoid
        # a model predicting nothing.

        # get all classes that have been observed so far
        all_observed_classes = {
            class_id
            for classes in self.observed_classes.values()
            for class_id in classes
        }

        # filter out all classes that have not been observed so far
        outputs = [
            output[
                torch.tensor(
                    [
                        class_id in self.observed_classes_per_exp[exp_id]
                        for class_id in all_observed_classes
                    ]
                )
            ]
            for exp_id, output in enumerate(outputs)
        ]

        # if a model has not seen a class, it will not predict this class, so we need to add a zero tensor
        # for this class
        outputs = [
            torch.cat(
                [
                    output,
                    torch.zeros(
                        len(all_observed_classes) - len(self.observed_classes[exp_id])
                    ),
                ]
            )
            for exp_id, output in enumerate(outputs)
        ]

        # stack all outputs and perform hard voting
        outputs = torch.stack(outputs)
        outputs = torch.mode(outputs, dim=0).values

        return outputs
