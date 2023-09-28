import numpy as np
from abc import ABC, abstractmethod, abstractproperty

from torch.utils.data import Dataset


class ContinualLearningDataset(ABC, Dataset):
    """Abstract class for continual learning datasets."""
    class SubDataset(Dataset):
        """Sub-dataset for a task."""
        def __init__(self, features: np.array, labels: np.array) -> None:
            super().__init__()
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def get_task(self, id: int):
        pass

    @abstractmethod
    def get_task_range(self, start: int, end: int):
        pass

    @abstractproperty
    def tasks_length(self):
        pass
