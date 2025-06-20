from enum import Enum
from typing import Any, Dict, NamedTuple, Union

import jax.numpy as jnp
import numpy as np


class FittingBackend(Enum):
    """Enumeration of available fitting backends."""

    NUMPY = "numpy"
    JAX = "jax"


class StatisticalTest(Enum):
    """Types of statistical tests available."""

    TWO_TAILED = "two_tailed"
    ONE_TAILED_UPPER = "one_tailed_upper"
    ONE_TAILED_LOWER = "one_tailed_lower"


# Default parameters
DEFAULT_CONFIG = {
    "min_obs": 5,
    "rcond": None,
    "confidence_level": 0.95,
    "significance_alpha": 0.05,
    "numerical_tolerance": 1e-10,
    "max_rank_deficiency_warning": True,
    "backend": FittingBackend.NUMPY,
}

# Critical values for common significance levels
CRITICAL_VALUES = {0.01: 2.576, 0.05: 1.96, 0.10: 1.645}


class FitResult(NamedTuple):
    """Enhanced container for fitting results with comprehensive metadata.

    Attributes
    ----------
    coeffs : Union[np.ndarray, jnp.ndarray]
        Parameter estimates. The coefficient ordering follows:
        - Polynomial terms: [constant, linear/factorial(1), quadratic/factorial(2), ...]
        - Periodic terms: [sin(2π*t/P1), cos(2π*t/P1), sin(2π*t/P2), cos(2π*t/P2), ...]
        - Step functions: [step1, step2, ...]
        - Exponential terms: [exp_decay1, exp_decay2, ...]
        - Logarithmic terms: [log_term1, log_term2, ...]
    mse : Union[np.ndarray, jnp.ndarray]
        Mean squared error per pixel
    Qxx : Union[np.ndarray, jnp.ndarray]
        Covariance matrix of parameter estimates
    residuals : Union[np.ndarray, jnp.ndarray]
        Model residuals (observed - fitted)
    n_valid : Union[np.ndarray, jnp.ndarray]
        Number of valid (non-NaN) observations used in fitting
    rank : Union[np.ndarray, jnp.ndarray]
        Effective rank of the design matrix
    metadata : Dict[str, Any]
        Additional fitting metadata including model specification and diagnostics

    """

    coeffs: Union[np.ndarray, jnp.ndarray]
    mse: Union[np.ndarray, jnp.ndarray]
    Qxx: Union[np.ndarray, jnp.ndarray]
    standard_errors: Union[np.ndarray, jnp.ndarray]
    residuals: Union[np.ndarray, jnp.ndarray]
    rank: Union[np.ndarray, jnp.ndarray]
    metadata: Dict[str, Any] = {}


class StatisticalResult(NamedTuple):
    """Container for comprehensive statistical analysis results."""

    coefficients: Union[np.ndarray, jnp.ndarray]
    standard_errors: Union[np.ndarray, jnp.ndarray]
    z_scores: Union[np.ndarray, jnp.ndarray]
    t_statistics: Union[np.ndarray, jnp.ndarray]
    p_values: Union[np.ndarray, jnp.ndarray]
    confidence_intervals_lower: Union[np.ndarray, jnp.ndarray]
    confidence_intervals_upper: Union[np.ndarray, jnp.ndarray]
    is_significant: Union[np.ndarray, jnp.ndarray]
    correlation_matrix: Union[np.ndarray, jnp.ndarray]
