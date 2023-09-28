import numpy as np


def to_categorical(y: int, num_classes: int):
    """Creates a 1-hot encoded array from a given integer
    Parameters
    ----------
    y: int
        Integer to be converted to 1-hot encoded array
    num_classes: int
        Number of classes in the dataset

    Returns
    -------
    np.ndarray
        1-hot encoded array
    """

    return np.eye(num_classes)[y]
