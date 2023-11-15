import numpy as np
import pandas as pd

from base_code.constants import DATASETS_PATH
from base_code.datasets.base import ContinualLearningDataset


class DryBeansDataset(ContinualLearningDataset):
    def __init__(self) -> None:
        super().__init__()
        dataset = pd.read_csv(DATASETS_PATH / "dry_beans.csv")
        self.features = dataset.drop(columns=["Class"]).values
        self.labels = dataset["Class"]
        self.__unique_labels = self.labels.unique()
        self.__num_classes = len(self.__unique_labels)
        self.labels_id = self.labels.map({
            label: i
            for i, label in enumerate(self.__unique_labels)
        })
        self.one_hot_labels = np.array([
            np.eye(self.__num_classes)[label_id]
            for label_id in self.labels_id
        ])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.one_hot_labels[idx],
        )

    @property
    def features_shape(self):
        return self.features.shape[1]

    @property
    def output_shape(self):
        return self.__num_classes

    @property
    def tasks_length(self):
        return self.__num_classes

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
        return self.SubDataset(
            self.features[self.labels_id == id],
            self.one_hot_labels[self.labels_id == id],
        )

    def get_task_range(self, start: int, end: int):
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
        return self.SubDataset(
            self.features[(self.labels_id >= start) & (self.labels_id < end)],
            self.one_hot_labels[(self.labels_id >= start) & (self.labels_id < end)],
        )
