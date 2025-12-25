"""Time series utility functions."""

import numpy as np
from numpy.typing import NDArray


def create_timeseries(
    x: NDArray[np.float32],
    cat: NDArray[np.float32],
    y: NDArray[np.float32],
    window: int = 10,
    n_out: int = 1,
    shift: int = 0,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Create sliding window time series for model input.

    Generates overlapping sequences from a time series for training
    sequence-to-sequence models.

    Args:
        X: Feature array of shape (T,) or (T, F) where T is time steps.
        cat: Categorical data array, one entry per time step.
        y: Target array of shape (T,).
        window: Number of historical time steps to include in each sample.
        n_out: Number of future time steps to predict.
        shift: Offset between input window end and target start.

    Returns:
        Tuple of (input_sequences, categorical_data, target_sequences):
            - input_sequences: Shape (N, window) or (N, window, F)
            - categorical_data: Shape (N,) with categorical data per sample
            - target_sequences: Shape (N, n_out)

    Raises:
        ValueError: If input length is insufficient for the requested window.

    Example:
        >>> X = np.arange(20).astype(np.float32)
        >>> cat = np.zeros(20)
        >>> y = np.arange(20).astype(np.float32)
        >>> x_seq, cat_seq, y_seq = create_timeseries(X, cat, y, window=5, n_out=2)
        >>> x_seq.shape
        (13, 5)
    """
    min_length = window + n_out + shift
    if len(x) < min_length:
        raise ValueError(
            f"Input length {len(x)} is too short for window={window}, "
            f"n_out={n_out}, shift={shift}. Minimum required: {min_length}"
        )

    _x, _cat, _y = [], [], []

    for i in range(len(x) - (window + n_out + shift)):
        _x.append(x[i : i + window])
        _cat.append(cat[i])
        _y.append(y[i + window + shift : i + window + shift + n_out])

    return np.asarray(_x), np.asarray(_cat), np.asarray(_y)
