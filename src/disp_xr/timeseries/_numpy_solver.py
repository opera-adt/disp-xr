import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.linalg

from ._constants import FitResult
from ._validate import validate_input_data

logger = logging.getLogger(__name__)


class LeastSquaresSolver:
    """Enhanced least squares solver with better numerical stability."""

    def __init__(
        self, rcond: Optional[float] = None, max_rank_deficiency_warning: bool = True
    ):
        self.rcond = rcond
        self.max_rank_deficiency_warning = max_rank_deficiency_warning

    def _compute_rank(self, matrix: np.ndarray) -> Tuple[int, float]:
        """Compute effective rank and condition number."""
        if self.rcond is None:
            rcond = np.finfo(matrix.dtype).eps * max(matrix.shape)
        else:
            rcond = self.rcond

        try:
            # Use SVD for more reliable rank computation
            _, s, _ = np.linalg.svd(matrix, full_matrices=False)
            tol = rcond * s[0] if len(s) > 0 and s[0] > 0 else rcond
            rank = np.sum(s > tol)
            condition_number = s[0] / s[-1] if len(s) > 0 and s[-1] > 0 else np.inf
            return rank, condition_number
        except np.linalg.LinAlgError:
            return 0, np.inf

    def solve_unweighted(
        self, A: np.ndarray, b: np.ndarray, full_stats: bool = False
    ) -> FitResult:
        """Solve unweighted least squares with enhanced numerical stability."""
        n_obs, n_params = A.shape
        n_pixels = b.shape[1]

        # Compute rank and condition number
        rank, condition_number = self._compute_rank(A)

        if rank < n_params and self.max_rank_deficiency_warning:
            warnings.warn(
                f"Matrix is rank deficient: rank {rank} < {n_params} "
                f"(condition number: {condition_number:.2e})",
                RuntimeWarning,
                stacklevel=3,
            )

        try:
            # Try QR decomposition first
            if rank == n_params:
                Q, R = np.linalg.qr(A, mode="reduced")
                z = Q.T @ b
                x = scipy.linalg.solve_triangular(R, z, check_finite=False)
                residuals = b - Q @ z

                # Compute covariance matrix
                R_inv = scipy.linalg.solve_triangular(
                    R, np.eye(n_params), check_finite=False
                )
                Qxx_base = R_inv @ R_inv.T
            else:
                # Fall back to SVD for rank-deficient case
                result = np.linalg.lstsq(A, b, rcond=self.rcond)
                x = result[0]
                residuals = b - A @ x

                # Compute pseudo-inverse for covariance
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                s_inv = np.where(s > (self.rcond or 1e-15) * s[0], 1 / s, 0)
                A_pinv = Vt.T @ np.diag(s_inv) @ U.T
                Qxx_base = A_pinv @ A_pinv.T

        except np.linalg.LinAlgError as e:
            logger.error(f"Linear algebra error in solver: {e}")
            # Return NaN results
            x = np.full((n_params, n_pixels), np.nan)
            residuals = np.full_like(b, np.nan)
            Qxx_base = np.full((n_params, n_params), np.nan)

        # Compute statistics
        dof = max(1, n_obs - rank)
        mse = np.sum(residuals**2, axis=0) / dof

        # Expand covariance matrix
        Qxx = mse[None, None, :] * Qxx_base[:, :, None]

        np.full(n_pixels, n_obs)

        metadata = {
            "condition_number": condition_number,
            "degrees_of_freedom": dof,
            "solver_method": "QR" if rank == n_params else "SVD",
        }
        # Get standard error
        variances = np.diagonal(Qxx, axis1=0, axis2=1).T
        std_errors = np.sqrt(np.maximum(variances, 0))

        if full_stats:
            return FitResult(
                x, mse, Qxx, std_errors, residuals, np.array(rank), metadata
            )

        else:
            return FitResult(x, mse, None, std_errors, None, np.array(rank), metadata)


def lscov_2d(
    A: np.ndarray,
    b: np.ndarray,
    weights: Optional[np.ndarray] = None,
    rcond: Optional[float] = None,
    full_stats: bool = False,
    **kwargs,
) -> FitResult:
    """Enhanced weighted least squares solver with better error handling."""
    # Input validation
    data, _, weights = validate_input_data(b, np.arange(b.shape[0]), weights)
    b = data.T

    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2D array, got {A.ndim}D")

    # Initialize solver
    solver = LeastSquaresSolver(rcond, kwargs.get("max_rank_deficiency_warning", True))

    # Route to appropriate solver
    if weights is None:
        return solver.solve_unweighted(A, b, full_stats=full_stats)
    else:
        # For weighted case, we'll transform the problem
        weights = np.asarray(weights, dtype=np.float64)

        if np.any(weights <= 0):
            raise ValueError("All weights must be positive")

        if weights.ndim == 1:
            sqrt_weights = np.sqrt(weights)
            A_weighted = A * sqrt_weights[:, np.newaxis]
            b_weighted = b * sqrt_weights[:, np.newaxis]
            return solver.solve_unweighted(
                A_weighted, b_weighted, full_stats=full_stats
            )
        else:
            # Handle variable weights pixel by pixel
            return _solve_variable_weights(A, b, weights, solver)


def _solve_variable_weights(
    A: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
    solver: LeastSquaresSolver,
    full_stats: bool = False,
) -> FitResult:
    """Handle variable weights by solving pixel by pixel."""
    n_obs, n_params = A.shape
    n_pixels = b.shape[1]

    # Pre-allocate results
    x = np.zeros((n_params, n_pixels))
    residuals = np.zeros((n_obs, n_pixels))
    mse = np.zeros(n_pixels)
    Qxx = np.zeros((n_params, n_params, n_pixels))
    n_valid = np.zeros(n_pixels)
    ranks = np.zeros(n_pixels, dtype=int)

    for i in range(n_pixels):
        w = weights[:, i] if weights.ndim == 2 else weights
        b_col = b[:, i : i + 1]

        sqrt_w = np.sqrt(w)
        A_w = A * sqrt_w[:, np.newaxis]
        b_w = b_col * sqrt_w[:, np.newaxis]

        result = solver.solve_unweighted(A_w, b_w)

        x[:, i : i + 1] = result.coeffs
        residuals[:, i : i + 1] = result.residuals / sqrt_w[:, np.newaxis]
        mse[i] = result.mse[0]
        Qxx[:, :, i] = result.Qxx[:, :, 0]
        n_valid[i] = np.sum(w)
        ranks[i] = result.rank

    rank = int(np.median(ranks))
    metadata = {"rank_distribution": ranks}

    # Get standard error
    variances = np.diagonal(Qxx, axis1=0, axis2=1).T
    std_errors = np.sqrt(np.maximum(variances, 0))

    if full_stats:
        return FitResult(x, mse, Qxx, std_errors, residuals, np.array(rank), metadata)
    else:
        return FitResult(x, mse, None, std_errors, None, np.array(rank), metadata)
