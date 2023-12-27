from collections import defaultdict
from typing import overload


class StoreLossBase:
    """Base class for storing loss values during training.

    The purpose of this class is to store custom loss with multiple components values during training.
    """

    def __init__(self):
        self.loss_store = defaultdict(list)

    @overload
    def store_loss(self, loss_value: float) -> None:
        """Store loss value.

        Parameters
        ----------
        loss_value : float
            Loss value to store.
        """
        ...

    @overload
    def store_loss(self, loss_value: float, key: str) -> None:
        """Store loss value.

        Parameters
        ----------
        loss_value : float
            Loss value to store.
        key : str
            Key to store loss value.
        """
        ...

    def store_loss(self, loss_value: float, key: str = "loss") -> None:
        """Store loss value.

        Parameters
        ----------
        loss_value : float
            Loss value to store.
        key : str
            Key to store loss value.
        """
        self.loss_store[key].append(loss_value)

    def get_loss(self, key: str = "loss"):
        """Get loss value.

        Parameters
        ----------
        key : str
            Key to get loss value.

        Returns
        -------
        float
            Loss value.
        """
        return self.loss_store[key]

    def get_loss_store(self):
        """Get loss store.

        Returns
        -------
        defaultdict
            Loss store.
        """
        return self.loss_store
