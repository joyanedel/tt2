import numpy as np

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline


def pipeline(x: np.ndarray):
    """Pipeline to normalize and standarize the data.

    Parameters
    ----------
    x : np.ndarray
        Data to be normalized and standarized.

    Returns
    -------
    Pipeline
        Pipeline to normalize and standarize the data.
    """
    return Pipeline(
        [
            ("normalizer", Normalizer()),
            ("standarizer", StandardScaler()),
        ]
    ).fit(x)
