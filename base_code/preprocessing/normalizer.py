import numpy as np
from torch import compile
from sklearn.preprocessing import Normalizer


@compile
def normalize(x: np.ndarray, *args, **kwargs) -> Normalizer:
    """Normalize the given array.

    Parameters
    ----------
    x : np.ndarray
        The array to normalize.

    Returns
    -------
    Normalizer
        The normalizer fitted on the given array.
    """
    return Normalizer(*args, **kwargs).fit(x)
