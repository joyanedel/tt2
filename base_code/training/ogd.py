from typing import Callable, Sequence, Union, TypeVar
from avalanche.core import BasePlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

CallbackResult = TypeVar("CallbackResult")


class OGDPlugin(SupervisedPlugin):
    """Orthogonal Gradient Descent plugin implementation that intercept training step"""

    def __init__(self):
        super().__init__()
        self.can_be_applied = False
        self.gradient_storage = []
        self.class_specific_gradient_storage = []
        self.orthonormal_basis = []

    def after_backward(self, strategy: SupervisedTemplate, *args, **kwargs) -> CallbackResult:
        g = self.__flatten_grad(strategy.model)
        self.class_specific_gradient_storage.append(g)

        proj_g = self.project_gradient_vector(g)
        g_tilde = g - proj_g

        # assign g_tilde as grad
        self.__update_model_grad(strategy.model, g_tilde)

    def after_training_exp(self, *args, **kwargs):
        self.gradient_storage.extend(self.class_specific_gradient_storage)
        self.class_specific_gradient_storage = []

        self.update_orthonormal_basis()
    
    @torch.no_grad
    def update_orthonormal_basis(self):
        """Update orthogonal basis for applying projection on incoming gradients"""
        # What does QR means in this context?
        # What is stack method? Stack method concatenates tensor into a new dimension, for this reason, all the tensors must be the same dimension
        # Why all the transpose operations? Seems related how QR understand the matrix we give as parameter
        # What is this method doing? This method take a list of storaged gradients and generate an orthogonal basis from that using QR factorization
        if len(self.gradient_storage) == 0:
            return
        
        q, _ = torch.linalg.qr(torch.stack(self.gradient_storage).T)
        self.orthonormal_basis = q.T
        self.can_be_applied = True
    
    @torch.no_grad
    def project_gradient_vector(self, g: torch.Tensor) -> torch.Tensor:
        """Return gradient ~g which is orthogonal to a specific precalculated basis
        
        If basis is empty, then return g without any operation applied
        """
        # What is g? its type? g is a gradient vector and its type is Tensor
        # What does view method do to a tensor: This methods change the shape of the tensor but retaining the data. -1 makes the method to infers the first dimension form original data and the following 1 is the other dimension
        # What is mm method from torch? mm stands for Matrix Multiplication and multiply the basis with the vector
        # This is (? x F) * (F x ¿) = (? x ¿) and transpose it to (¿ x ?)
        # Then res = (¿ x ?) * (? x F) -> (¿ x F) and transpose it to (F x ¿)
        # Why all the transpose operations?
        # What is this method doing?
        if not self.can_be_applied:
            return g

        mid = (torch.mm(self.orthonormal_basis, g.view(-1, 1))).T
        res = (torch.mm(mid, self.orthonormal_basis)).T

        return res.view(-1)

    def __flatten_grad(self, model: Module) -> torch.Tensor:
        grads = [
            param.grad.view(-1)
            for _, param in model.named_parameters()
        ]

        return torch.cat(grads)
    
    def __update_model_grad(self, model: Module, grads: torch.Tensor):
        index = 0
        state_dict = model.state_dict(keep_vars=True)

        for param in state_dict.keys():
            # ignore batchnorm params
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param].grad = grads[index:index+param_count].view(param_shape).clone()
            index += param_count
        
        model.load_state_dict(state_dict)


class OGD(SupervisedTemplate):
    """Orthogonal Gradient Descent strategy implementation"""

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Union[int, None]= 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Union[Sequence[BasePlugin], None] = None,
        evaluator: Union[EvaluationPlugin, Callable[[], EvaluationPlugin]] = ...,
        eval_every=-1,
        **base_kwargs,
    ):
        ogd = OGDPlugin()

        if plugins is None:
            plugins = []
        plugins.append(ogd)

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
            **base_kwargs
        )