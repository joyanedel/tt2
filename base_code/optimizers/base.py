from abc import ABC, abstractmethod


class StatefulOptimizer(ABC):
    "Base class for stateful optimizers"

    @abstractmethod
    def update(self):
        pass
