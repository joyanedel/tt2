import numpy as np
from torch import compile
from sklearn.preprocessing import StandardScaler


@compile
def standarize(x: np.ndarray) -> StandardScaler:
    """Standarize the given array.

    Parameters
    ----------
    x : np.ndarray
        The array to normalize.

    Returns
    -------
    StandardScaler
        The standarizer fitted on the given array.
    """
    return StandardScaler().fit(x)
