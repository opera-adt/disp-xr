from functools import partial
from typing import Any, List, Optional, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from ._constants import FitResult
from ._design_matrix import create_design_matrix

# NOTE need to refine this code to implement it
# jax.config.update("jax_enable_x64", False)


@jax.jit
def jax_solve_linear_system(
    A: jnp.ndarray,
    b: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    tol: float = 1e-10,
) -> FitResult:
    """Enhanced JAX linear system solver with better numerical stability."""
    n_obs, n_params = A.shape

    # Handle weights
    weights = (
        jnp.ones((n_obs, 1), dtype=b.dtype)
        if weights is None
        else weights.reshape((n_obs, 1))
    )
    sqrt_weights = jnp.sqrt(jnp.maximum(weights, 0.0))
    A_weighted = A * sqrt_weights
    b_weighted = b * sqrt_weights

    # QR decomposition
    Q, R = jnp.linalg.qr(A_weighted, mode="reduced")
    z = Q.T @ b_weighted

    # Rank
    r_diag = jnp.abs(jnp.diag(R))
    rank_tol = tol * jnp.maximum(jnp.max(r_diag), tol)
    rank = jnp.sum(r_diag > rank_tol)

    # Solve
    R_reg = R + jnp.eye(n_params, dtype=R.dtype) * tol
    x = jax.scipy.linalg.solve_triangular(R_reg, z)

    # Residuals
    fitted_weighted = Q @ z
    residuals_weighted = b_weighted - fitted_weighted
    residuals_weighted / jnp.maximum(sqrt_weights, tol)

    n_effective = jnp.sum(weights)
    dof = jnp.maximum(1.0, n_effective - rank)
    mse = jnp.nansum(residuals_weighted**2, axis=0) / dof

    # Try to compute Qxx
    def compute_qxx_valid(_):
        R_inv = jax.scipy.linalg.solve_triangular(R, jnp.eye(n_params, dtype=R.dtype))
        return R_inv @ R_inv.T

    def compute_qxx_fallback(_):
        R_pinv = jnp.linalg.pinv(R)
        return R_pinv @ R_pinv.T

    Qxx_base = lax.cond(
        jnp.linalg.cond(R) < 1 / tol,
        compute_qxx_valid,
        compute_qxx_fallback,
        operand=None,
    )

    Qxx = mse[None, None, :] * Qxx_base[:, :, None]
    variances = jnp.trace(Qxx, axis1=0, axis2=1)  # shape (n_fits,)
    std_errors = jnp.sqrt(jnp.maximum(variances, 0.0))

    # Return full or compact stats
    return FitResult(x, mse, None, std_errors, None, rank)


@jax.jit
def _solve_with_missing_data_jax(
    A: jnp.ndarray,
    b: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    min_obs: int = 5,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Solve linear system with missing data and return full statistics.

    Parameters
    ----------
    A : jnp.ndarray
        Design matrix (n_obs, n_params)
    b : jnp.ndarray
        Observations (n_obs,) - may contain NaN
    weights : jnp.ndarray, optional
        Observation weights (n_obs,)
    min_obs : int
        Minimum number of valid observations

    Returns
    -------
    x : jnp.ndarray
        Solution (n_params,)
    mse : float
        Mean squared error
    Qxx : jnp.ndarray
        Covariance matrix (n_params, n_params)
    residuals : jnp.ndarray
        Residuals (n_obs,)
    n_valid : int
        Number of valid observations used
    rank : int
        Effective rank of design matrix

    """
    n_obs, n_params = A.shape
    n_pixels = b.shape[0]

    # Create validity mask
    valid_mask = ~jnp.isnan(b)
    n_valid = jnp.sum(valid_mask)

    # Replace NaN values with zeros
    b_clean = jnp.where(valid_mask, b, 0.0)

    # Create effective weights
    if weights is None:
        eff_weights = valid_mask.astype(jnp.float32)
    else:
        eff_weights = jnp.where(valid_mask, weights, 0.0)

    def solve_case():
        x, mse, Qxx, residuals, rank = jax_solve_linear_system(A, b_clean, eff_weights)
        # Mask residuals for invalid observations
        masked_residuals = jnp.where(valid_mask, residuals, jnp.nan)
        return x, mse, Qxx, masked_residuals, n_valid, rank

    def insufficient_data_case():
        nan_coeffs = jnp.full((n_params, n_pixels), jnp.nan)
        nan_mse = jnp.full((n_pixels,), jnp.nan)
        nan_Qxx = jnp.full((n_params, n_params, n_pixels), jnp.nan)
        nan_residuals = jnp.full((n_obs, n_pixels), jnp.nan)
        zero_rank = jnp.array(0)
        return nan_coeffs, nan_mse, nan_Qxx, nan_residuals, n_valid, zero_rank

    return lax.cond(n_valid >= min_obs, solve_case, insufficient_data_case)


@partial(jax.jit, static_argnames=["min_obs"])
def fit_timeseries_pixelwise_jax(
    A: jnp.ndarray,
    pixel_data: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    min_obs: int = 5,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Fit time series model to a single pixel with full statistics.

    Parameters
    ----------
    A : jnp.ndarray
        Design matrix (n_time, n_params)
    pixel_data : jnp.ndarray
        Time series data for one pixel (n_time,)
    weights : jnp.ndarray, optional
        Observation weights (n_time,)
    min_obs : int
        Minimum observations required for fitting

    Returns
    -------
    coeffs : jnp.ndarray
        Model coefficients (n_params,)
    mse : float
        Mean squared error
    Qxx : jnp.ndarray
        Covariance matrix (n_params, n_params)
    residuals : jnp.ndarray
        Residuals (n_time,)
    n_valid : int
        Number of valid observations
    rank : int
        Effective rank

    """
    return _solve_with_missing_data_jax(A, pixel_data, weights, min_obs)


# Note gives me wrong shape
@partial(jax.jit, static_argnames=["min_obs"])
def fit_timeseries_2d_jax(
    A: jnp.ndarray,
    B: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    min_obs: int = 5,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Fit time series model to 2D array with full statistics using vectorized JAX.

    Parameters
    ----------
    A : jnp.ndarray
        Design matrix (n_time, n_params)
    B : jnp.ndarray
        Time series data (n_time, n_pixels)
    weights : jnp.ndarray, optional
        Observation weights (n_time,) or (n_time, n_pixels)
    min_obs : int
        Minimum observations required per pixel

    Returns
    -------
    coeffs : jnp.ndarray
        Model coefficients (n_params, n_pixels)
    mse : jnp.ndarray
        Mean squared error per pixel (n_pixels,)
    Qxx : jnp.ndarray
        Covariance matrices (n_params, n_params, n_pixels)
    residuals : jnp.ndarray
        Residuals (n_time, n_pixels)
    n_valid : jnp.ndarray
        Number of valid observations per pixel (n_pixels,)
    rank : jnp.ndarray
        Effective rank per pixel (n_pixels,)

    """
    if weights is None:
        # No weights case
        def fit_fn(pixel: Any, w: Any = ...) -> Any:
            return fit_timeseries_pixelwise_jax(A, pixel, w, min_obs)  # type: ignore

        results = jax.vmap(fit_fn, in_axes=1, out_axes=(1, 0, 2, 1, 0, 0))(B)

    elif weights.ndim == 1:
        # Same weights for all pixels
        def fit_fn(pixel: Any, w: Any = ...) -> Any:
            return fit_timeseries_pixelwise_jax(A, pixel, w, min_obs)  # type: ignore

        results = jax.vmap(fit_fn, in_axes=1, out_axes=(1, 0, 2, 1, 0, 0))(B)

    else:
        # Different weights per pixel
        def fit_fn(pixel: Any, w: Any = ...) -> Any:
            return fit_timeseries_pixelwise_jax(A, pixel, w, min_obs)  # type: ignore

        results = jax.vmap(fit_fn, in_axes=(1, 1), out_axes=(1, 0, 2, 1, 0, 0))(
            B, weights
        )

    return results


def fit_timeseries_3d_jax(
    data: jnp.ndarray,
    dates: np.ndarray,
    polynomial_degree: int = 1,
    periods: Optional[List[float]] = None,
    weights: Optional[jnp.ndarray] = None,
    min_obs: int = 5,
) -> FitResult:
    """Fit time series model to 3D data with comprehensive statistics using JAX.

    Parameters
    ----------
    data : jnp.ndarray
        Input data (n_time, n_height, n_width)
    dates : np.ndarray
        Time points (n_time,)
    polynomial_degree : int
        Polynomial degree for trend
    periods : List[float], optional
        List of periods for seasonal terms
    weights : jnp.ndarray, optional
        Observation weights
    min_obs : int
        Minimum observations per pixel

    Returns
    -------
    FitResult
        Named tuple containing:
        - coeffs : jnp.ndarray (n_params, n_height, n_width)
        - mse : jnp.ndarray (n_height, n_width)
        - Qxx : jnp.ndarray (n_params, n_params, n_height, n_width)
        - residuals : jnp.ndarray (n_time, n_height, n_width)
        - n_valid : jnp.ndarray (n_height, n_width)
        - rank : jnp.ndarray (n_height, n_width)

    """
    n_time, n_height, n_width = data.shape

    # Create design matrix using NumPy, then convert to JAX
    A_np = create_design_matrix(dates, polynomial_degree, periods)
    A_jax = jnp.array(A_np, dtype=jnp.float32)
    n_params = A_jax.shape[1]

    # Reshape data to 2D for processing
    data_2d = data.reshape(n_time, n_height * n_width)

    # Handle weights
    if weights is not None:
        if weights.ndim == 1:
            weights_2d = weights
        elif weights.ndim == 3:
            weights_2d = weights.reshape(n_time, n_height * n_width)
        else:
            raise ValueError(
                (
                    f"Weights shape {weights.shape}"
                    f"not compatible with data shape {data.shape}"
                )
            )
    else:
        weights_2d = None

    # Fit using 2D JAX function
    (coeffs_2d, mse_1d, Qxx_3d, residuals_2d, n_valid_1d, rank_1d) = (
        fit_timeseries_2d_jax(A_jax, data_2d, weights_2d, min_obs)
    )

    # Reshape results back to spatial dimensions
    coeffs = coeffs_2d.reshape(n_params, n_height, n_width)
    mse = mse_1d.reshape(n_height, n_width)
    Qxx = Qxx_3d.reshape(n_params, n_params, n_height, n_width)
    residuals = residuals_2d.reshape(n_time, n_height, n_width)
    n_valid_1d.reshape(n_height, n_width)
    rank = rank_1d.reshape(n_height, n_width)

    return FitResult(
        coeffs=coeffs,
        mse=mse,
        Qxx=Qxx,
        standard_errors=None,
        residuals=residuals,
        rank=rank,
    )
