from typing import Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np


def validate_input_data(
    data: Union[np.ndarray, jnp.ndarray],
    dates: np.ndarray,
    weights: Optional[Union[np.ndarray, jnp.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Validate and standardize input data.

    Parameters
    ----------
    data : array-like
        Input time series data
    dates : array-like
        Time points
    weights : array-like, optional
        Observation weights

    Returns
    -------
    data_clean : np.ndarray
        Validated data array
    dates_clean : np.ndarray
        Validated dates array
    weights_clean : np.ndarray or None
        Validated weights array

    Raises
    ------
    ValueError
        If input data is invalid

    """
    # Convert to numpy arrays
    data = np.asarray(data, dtype=np.float64)
    dates = np.asarray(dates)

    # Validate shapes
    if data.ndim == 0:
        raise ValueError("Data must be at least 1-dimensional")
    if data.ndim > 3:
        raise ValueError(f"Data must be at most 3-dimensional, got {data.ndim}D")

    n_time = data.shape[0]
    if len(dates) != n_time:
        raise ValueError(
            f"Length mismatch: data has {n_time} time points, dates has {len(dates)}"
        )

    # Validate weights if provided
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim > data.ndim:
            raise ValueError(
                f"Weights dim. ({weights.ndim}) exceeds data dim. ({data.ndim})"
            )

        # Check weight shapes
        if weights.ndim == 1 and len(weights) != n_time:
            raise ValueError(
                f"1D weights must have length {n_time}, got {len(weights)}"
            )
        elif weights.ndim > 1 and weights.shape[0] != n_time:
            raise ValueError(
                f"Multi-dimensional weights must have {n_time} time points"
            )

        if np.any(weights < 0):
            raise ValueError("All weights must be non-negative")
        if np.all(weights == 0):
            raise ValueError("At least some weights must be positive")

    return data.T, dates, weights
