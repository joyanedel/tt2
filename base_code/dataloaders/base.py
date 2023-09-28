from torch.utils.data import DataLoader
from base_code.datasets.base import ContinualLearningDataset


class ContinualLearningDataLoader(DataLoader):
    """DataLoader for continual learning."""

    def __init__(
        self, dataset: ContinualLearningDataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 1
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.shuffle = shuffle

    def get_task(self, id: int):
        """Get the task with the given id.

        Parameters
        ----------
        id : int
            Task id.

        Returns
        -------
        ContinualLearningDataset.TaskInfo
            Task information.
        """
        return DataLoader(
            self.dataset.get_task(id),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_tasks_length(self):
        return self.dataset.tasks_length

    def get_task_range(self, start: int, end: int):
        return DataLoader(
            self.dataset.get_task_range(start, end),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
