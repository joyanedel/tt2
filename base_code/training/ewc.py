import torch
from torch import nn
from torch.optim import Optimizer

from base_code.dataloaders.base import ContinualLearningDataLoader
from base_code.losses.base import StatefulLoss


def train(
    model: nn.Module,
    dataloader: ContinualLearningDataLoader,
    loss_fn: StatefulLoss,
    optimizer: Optimizer,
):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y, model)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    loss_fn.update(model, dataloader)


def test(model: nn.Module, dataloader: ContinualLearningDataLoader, loss_fn: StatefulLoss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y, model).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
