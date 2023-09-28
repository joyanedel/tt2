from copy import deepcopy
from collections import defaultdict
from torch import Tensor
from torch.autograd import Variable as variable
from torch.nn import functional as F, Module
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter
from torch.utils.data import dataloader

from typing import Dict, Optional

from base_code.losses.base import StatefulLoss


class EWC(CrossEntropyLoss, StatefulLoss):
    """Elastic Weight Consolidation (EWC) loss."""

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        importance: float = 1,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = 'mean',
        label_smoothing: float = 0
    ) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        self.importance = importance
        self._means = defaultdict(Tensor) # TODO: check if this is the right way to initialize # noqa
        self._precision_matrices = defaultdict(Tensor) # TODO: check if this is the right way to initialize # noqa

    def _diag_fisher(self, params: Dict[str, Parameter], dataset: dataloader):
        precision_matrices = {}

        for n, p in params.items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        for input in dataset:
            self.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in params.items():
                precision_matrices[n].data += p.grad.data ** 2 / len(dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def _get_means(self, params: Dict[str, Parameter]):
        return {n: variable(p.data) for n, p in params.items()}

    def penalty(self, model: Module) -> Tensor:
        loss = 0

        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()

        return loss

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        model: Module
    ) -> Tensor:
        loss = super().forward(input, target)
        ewc_loss = loss + self.importance * self.penalty(model)

        return ewc_loss

    def update(
        self,
        model: Module,
        dataset: dataloader
    ) -> None:
        with model.eval() as model_eval:
            params = deepcopy({n: p for n, p in model_eval.named_parameters() if p.requires_grad})
            # update means
            self._means.update(self._get_means(params))

            # update precision matrices
            self._precision_matrices.update(self._diag_fisher(params, dataset))


# class EWC(object):
#     def __init__(self, model: nn.Module, dataset: list):

#         self.model = model
#         self.dataset = dataset

#         self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
#         self._means = {}
#         self._precision_matrices = self._diag_fisher()

#         for n, p in deepcopy(self.params).items():
#             self._means[n] = variable(p.data)

#     def _diag_fisher(self):
#         precision_matrices = {}
#         for n, p in deepcopy(self.params).items():
#             p.data.zero_()
#             precision_matrices[n] = variable(p.data)

#         self.model.eval()
#         for input in self.dataset:
#             self.model.zero_grad()
#             input = variable(input)
#             output = self.model(input).view(1, -1)
#             label = output.max(1)[1].view(-1)
#             loss = F.nll_loss(F.log_softmax(output, dim=1), label)
#             loss.backward()

#             for n, p in self.model.named_parameters():
#                 precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

#         precision_matrices = {n: p for n, p in precision_matrices.items()}
#         return precision_matrices

#     def penalty(self, model: nn.Module):
#         loss = 0
#         for n, p in model.named_parameters():
#             _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
#             loss += _loss.sum()
#         return loss
