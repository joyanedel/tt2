import torch
from torch.nn import Module
from torch.optim import Optimizer

from base_code.dataloaders.base import ContinualLearningDataLoader
from base_code.losses.base import StatefulLoss


def train(
    model: Module,
    dataloader: ContinualLearningDataLoader,
    loss_fn: Module,
    optimizer: Optimizer
):
    """Train the model on the given dataset.

    Parameters
    ----------
    dataloader : DataLoader
        The dataset to train the model on.
    model : Module
        The model to train.
    loss_fn : callable
        The loss function to use.
    optimizer : Optimizer
        The optimizer to use.

    Returns
    -------
    None
    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y, model)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    if isinstance(loss_fn, StatefulLoss):
        loss_fn.update(model, optimizer.param_groups[0]['params'])

def test(
    model: Module,
    dataloader: ContinualLearningDataLoader,
    loss_fn: callable
):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
