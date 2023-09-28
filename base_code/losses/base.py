from abc import ABC, abstractmethod


class StatefulLoss(ABC):
    """Abstract base class for stateful losses."""

    @abstractmethod
    def update(self, model, params):
        pass
