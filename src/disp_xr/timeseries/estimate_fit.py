from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from ._design_matrix import (
    create_design_matrix,
    get_coefficient_names,
    infer_parameters_from_names,
)
from ._numpy_solver import lscov_2d


def fit_timeseries(
    dxr: xr.DataArray,
    weights: Optional[np.ndarray] = None,
    polynomial_degree: int = 1,
    periods: Optional[List[float]] = None,
    step_dates: Optional[List[str]] = None,
    exponential_terms: Optional[Dict[str, List[int]]] = None,
    logarithmic_terms: Optional[Dict[str, List[int]]] = None,
    reference_index: int = 0,
) -> xr.Dataset:
    """Fit a time series using a design matrix.

    Parameters
    ----------
    dxr : xr.DataArray
        Input time series data with shape (time, y, x).
    weights : np.ndarray, optional
        Optional weights for each observation.
    polynomial_degree : int
        Degree of polynomial terms to include in the fit.
    periods : list of float, optional
        Periodic components (e.g., annual = 12, semiannual = 6).
    step_dates : list of str, optional
        Dates where step changes (offsets) occur.
    exponential_terms : dict, optional
        Exponential terms as a dict with 'start_dates' and 'decay_constants'.
    logarithmic_terms : dict, optional
        Logarithmic terms as a dict with 'start_dates' and 'powers'.
    reference_index : int
        Index in time series to use as the reference epoch.

    Returns
    -------
    xr.Dataset
        Dataset containing coefficients, standard errors, and mean squared error (MSE).

    """
    # Design matrix and coefficient names
    A = create_design_matrix(
        dxr.time.values,
        polynomial_degree,
        periods,
        step_dates,
        exponential_terms,
        logarithmic_terms,
        reference_index,
    )
    coeff_names = get_coefficient_names(
        polynomial_degree, periods, step_dates, exponential_terms, logarithmic_terms
    )

    # Flatten spatial dimensions
    n_time, n_height, n_width = dxr.shape
    n_params = A.shape[1]
    data_2d = dxr.values.reshape(n_time, -1, order="C")  # (time, pixels)

    # Fit model
    results = lscov_2d(A, data_2d, weights=weights)

    # Package results in an xarray.Dataset
    ds = xr.Dataset(
        {
            "coefficients": (
                ["coefficient", "y", "x"],
                results.coeffs.reshape(n_params, n_height, n_width),
            ),
            "standard_errors": (
                ["coefficient", "y", "x"],
                results.standard_errors.reshape(n_params, n_height, n_width),
            ),
            "mse": (["y", "x"], results.mse.reshape(n_height, n_width)),
        },
        coords={
            "coefficient": coeff_names,
            "y": dxr.y.values if "y" in dxr.dims else np.arange(n_height),
            "x": dxr.x.values if "x" in dxr.dims else np.arange(n_width),
        },
        attrs={"description": "Time series fit results with statistical metrics."},
    )

    return ds


def create_fit_template(
    n_params: int, y: np.ndarray, x: np.ndarray, coeff_names: List[str]
) -> xr.Dataset:
    """Create an empty template xarray.Dataset for time series fit results.

    Parameters
    ----------
    n_params : int
        Number of parameters (design matrix columns).
    y : np.ndarray
        Y-coordinate values (1D).
    x : np.ndarray
        X-coordinate values (1D).
    coeff_names : list of str
        Names of coefficients corresponding to each design matrix column.

    Returns
    -------
    xr.Dataset
        Dataset filled with NaNs and the correct dimensions and coordinates.

    """
    n_height = y.size
    n_width = x.size

    return xr.Dataset(
        {
            "coefficients": (
                ["coefficient", "y", "x"],
                np.full((n_params, n_height, n_width), np.nan),
            ),
            "standard_errors": (
                ["coefficient", "y", "x"],
                np.full((n_params, n_height, n_width), np.nan),
            ),
            "mse": (["y", "x"], np.full((n_height, n_width), np.nan)),
        },
        coords={
            "coefficient": coeff_names,
            "y": y,
            "x": x,
        },
        attrs={
            "description": (
                "Template for time series fit results with statistical metrics."
            )
        },
    )


def fit_timeseries_block(
    ds: xr.DataArray,
    weights: Optional[np.ndarray] = None,
    polynomial_degree: int = 1,
    periods: Optional[List[float]] = None,
    step_dates: Optional[List[str]] = None,
    exponential_terms: Optional[Dict[str, List[int]]] = None,
    logarithmic_terms: Optional[Dict[str, List[int]]] = None,
    reference_index: int = 0,
) -> xr.Dataset:
    """Apply pixelwise time series fitting to a chunked xarray.DataArray.

    Parameters
    ----------
    ds : xr.DataArray
        Time series data with shape (time, y, x).
    weights : np.ndarray, optional
        Optional weights for observations.
    polynomial_degree : int
        Degree of polynomial to fit.
    periods : list of float, optional
        List of periodic terms (e.g., [12] for annual).
    step_dates : list of str, optional
        List of dates for step discontinuities.
    exponential_terms : dict, optional
        Dict of exponential decay start dates and rates.
    logarithmic_terms : dict, optional
        Dict of log start dates and decay constants.
    reference_index : int
        Time index to be used as reference.

    Returns
    -------
    xr.Dataset
        Dataset with fit coefficients and statistics per pixel.

    """
    coefficient_kwargs = {
        "polynomial_degree": polynomial_degree,
        "periods": periods,
        "step_dates": step_dates,
        "exponential_terms": exponential_terms,
        "logarithmic_terms": logarithmic_terms,
    }

    # Get coefficient names and create output template
    coeff_names = get_coefficient_names(**coefficient_kwargs)  # type: ignore

    # Then add reference_index for actual fitting
    fitting_kwargs = {**coefficient_kwargs, "reference_index": reference_index}

    template = create_fit_template(len(coeff_names), ds.y, ds.x, coeff_names)
    template = template.chunk({"y": ds.chunks[1], "x": ds.chunks[2]})

    # Apply map_blocks
    return ds.map_blocks(
        fit_timeseries, kwargs={"weights": weights, **fitting_kwargs}, template=template
    )


def get_predicted(coeffs: xr.DataArray, obs_dates: np.ndarray) -> xr.DataArray:
    """Compute predicted values from regression coefficients and observation dates.

    Parameters
    ----------
    coeffs : xr.DataArray or None
        A DataArray of regression coefficients with a 'coefficient' dimension.
        The coefficient names are used to infer the model structure
        (e.g., intercept, linear trend, seasonal harmonics).
    obs_dates : np.ndarray
        A NumPy array of observation dates as np.datetime64, used to construct
        the design matrix for prediction.

    Returns
    -------
    predicted : xr.DataArray
        The predicted values computed as the dot product between the design matrix
        and the coefficient array. Output has 'time' dimension and any remaining
        spatial dimensions from `coeffs`.

    """
    # Handle None coeffs
    if coeffs is None:
        return xr.DataArray(
            np.full(len(obs_dates), np.nan), dims=["time"], coords={"time": obs_dates}
        )

    # Handle empty coefficient arrays
    if coeffs.sizes.get("coefficient", 0) == 0:
        empty_coords = {k: v for k, v in coeffs.coords.items() if k != "coefficient"}
        empty_coords["time"] = obs_dates
        empty_dims = ["time"] + [d for d in coeffs.dims if d != "coefficient"]
        empty_shape = (len(obs_dates),) + tuple(
            coeffs.sizes[d] for d in coeffs.dims if d != "coefficient"
        )
        return xr.DataArray(
            np.full(empty_shape, np.nan, dtype=coeffs.dtype),
            dims=empty_dims,
            coords=empty_coords,
        )

    # Infer fitting parameters and create design matrix
    fitting_kwargs = infer_parameters_from_names(coeffs.coefficient.values)
    A = create_design_matrix(obs_dates, **fitting_kwargs)

    # Convert to xarray and compute predictions using element-wise operations
    A_da = xr.DataArray(
        A,
        dims=["time", "coefficient"],
        coords={"time": obs_dates, "coefficient": coeffs.coefficient},
    )
    predicted = xr.dot(A_da, coeffs, dims="coefficient")

    return predicted


def sincos_to_amplitude_phase(
    A_cos: np.ndarray, A_sin: np.ndarray, period: float = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sine and cosine coefficients to amplitude and phase representation.

    Parameters
    ----------
    A_cos : float or np.ndarray
        Coefficient of the cosine term.
    A_sin : float or np.ndarray
        Coefficient of the sine term.
    period : float, default 12
        The period of the harmonic function. Must be positive.

    Returns
    -------
    amplitude : float or np.ndarray
        The amplitude of the harmonic function. Always non-negative.
    phase : float or np.ndarray
        The phase shift in units of the period, in range [0, period).

    """
    amplitude = np.sqrt(A_cos**2 + A_sin**2)
    phase = (period / (2 * np.pi)) * np.arctan2(-A_sin, A_cos)
    return amplitude, phase % period
