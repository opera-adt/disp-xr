import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.special import factorial

from ._datetime import _date_str_to_decimal_year, _dates_to_decimal_years


class DesignMatrixBuilder:
    """Builder class for constructing design matrices with validation and caching."""

    def __init__(self, dates: np.ndarray, reference_index: int = 0):
        self.dates = dates
        self.reference_index = reference_index
        self.time_years = _dates_to_decimal_years(dates, reference_index)
        self._cache: dict[str, Any] = {}

    def add_polynomial_terms(self, degree: int) -> np.ndarray:
        """Add polynomial terms with factorial normalization."""
        if degree < 0:
            raise ValueError("Polynomial degree must be non-negative")

        cache_key = f"poly_{degree}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        columns = []
        for i in range(degree + 1):
            if i == 0:
                columns.append(np.ones_like(self.time_years))
            else:
                columns.append((self.time_years**i) / factorial(i))

        result = np.column_stack(columns)
        self._cache[cache_key] = result
        return result

    def add_periodic_terms(self, periods: List[float]) -> np.ndarray:
        """Add periodic terms (sin/cos pairs)."""
        if not periods:
            return np.empty((len(self.time_years), 0))

        cache_key = f"periodic_{hash(tuple(periods))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        columns = []
        for period in periods:
            if period <= 0:
                raise ValueError(f"Period must be positive, got {period}")
            freq = 2 * np.pi / period
            columns.extend(
                [np.sin(freq * self.time_years), np.cos(freq * self.time_years)]
            )

        result = np.column_stack(columns)
        self._cache[cache_key] = result
        return result

    def add_step_terms(self, step_dates: List[str]) -> np.ndarray:
        """Add step function terms."""
        if not step_dates:
            return np.empty((len(self.time_years), 0))

        columns = []
        for step_date in step_dates:
            step_time = _date_str_to_decimal_year(step_date)
            step_relative = step_time - self.time_years[self.reference_index]
            columns.append((self.time_years >= step_relative).astype(float))

        return np.column_stack(columns)

    def add_exponential_terms(
        self, exponential_terms: Dict[str, List[int]]
    ) -> np.ndarray:
        """Add exponential decay terms."""
        if not exponential_terms:
            return np.empty((len(self.time_years), 0))

        columns = []
        for onset_date, char_times in exponential_terms.items():
            onset_time = (
                _date_str_to_decimal_year(onset_date)
                - self.time_years[self.reference_index]
            )
            for tau_days in char_times:
                if tau_days <= 0:
                    raise ValueError(
                        f"Characteristic time must be positive, got {tau_days}"
                    )
                tau_years = tau_days / 365.25
                dt = self.time_years - onset_time
                mask = dt >= 0
                exp_term = np.zeros_like(self.time_years)
                exp_term[mask] = 1 - np.exp(-dt[mask] / tau_years)
                columns.append(exp_term)

        return np.column_stack(columns)

    def add_logarithmic_terms(
        self, logarithmic_terms: Dict[str, List[int]]
    ) -> np.ndarray:
        """Add logarithmic terms."""
        if not logarithmic_terms:
            return np.empty((len(self.time_years), 0))

        columns = []
        for onset_date, char_times in logarithmic_terms.items():
            onset_time = (
                _date_str_to_decimal_year(onset_date)
                - self.time_years[self.reference_index]
            )
            for tau_days in char_times:
                if tau_days <= 0:
                    raise ValueError(
                        f"Characteristic time must be positive, got {tau_days}"
                    )
                tau_years = tau_days / 365.25
                dt = self.time_years - onset_time
                mask = dt >= 0
                log_term = np.zeros_like(self.time_years)
                with np.errstate(invalid="ignore", divide="ignore"):
                    valid_dt = dt[mask]
                    log_term[mask] = np.log1p(valid_dt / tau_years)
                columns.append(log_term)

        return np.column_stack(columns)


def create_design_matrix(
    dates: np.ndarray,
    polynomial_degree: int = 1,
    periods: Optional[List[float]] = None,
    step_dates: Optional[List[str]] = None,
    exponential_terms: Optional[Dict[str, List[int]]] = None,
    logarithmic_terms: Optional[Dict[str, List[int]]] = None,
    reference_index: int = 0,
) -> np.ndarray:
    """Create design matrix for time series analysis with enhanced validation.

    Parameters
    ----------
    dates : np.ndarray
        Array of dates (datetime64 format)
    polynomial_degree : int, default=1
        Polynomial degree (0=offset, 1=linear, 2=quadratic, etc.)
    periods : List[float], optional
        Periods in years for periodic terms
    step_dates : List[str], optional
        Step function onset dates in 'YYYYMMDD' format
    exponential_terms : Dict[str, List[int]], optional
        Exponential terms: {'YYYYMMDD': [char_time_days, ...]}
    logarithmic_terms : Dict[str, List[int]], optional
        Logarithmic terms: {'YYYYMMDD': [char_time_days, ...]}
    reference_index : int, default=0
        Index of reference date

    Returns
    -------
    np.ndarray
        Design matrix A of shape (n_obs, n_params)

    """
    # Set defaults
    periods = periods or []
    step_dates = step_dates or []
    exponential_terms = exponential_terms or {}
    logarithmic_terms = logarithmic_terms or {}

    # Build design matrix
    builder = DesignMatrixBuilder(dates, reference_index)

    matrix_parts = []

    # Add polynomial terms
    poly_matrix = builder.add_polynomial_terms(polynomial_degree)
    matrix_parts.append(poly_matrix)

    # Add periodic terms
    if periods:
        periodic_matrix = builder.add_periodic_terms(periods)
        matrix_parts.append(periodic_matrix)

    # Add step terms
    if step_dates:
        step_matrix = builder.add_step_terms(step_dates)
        matrix_parts.append(step_matrix)

    # Add exponential terms
    if exponential_terms:
        exp_matrix = builder.add_exponential_terms(exponential_terms)
        matrix_parts.append(exp_matrix)

    # Add logarithmic terms
    if logarithmic_terms:
        log_matrix = builder.add_logarithmic_terms(logarithmic_terms)
        matrix_parts.append(log_matrix)

    if not matrix_parts:
        raise ValueError("At least one time function must be specified")

    # Combine all parts
    design_matrix = np.concatenate(matrix_parts, axis=1)

    # Check for numerical issues
    if np.any(~np.isfinite(design_matrix)):
        warnings.warn("Design matrix contains non-finite values", RuntimeWarning)

    return design_matrix


def get_coefficient_names(
    polynomial_degree: int = 1,
    periods: Optional[List[float]] = None,
    step_dates: Optional[List[str]] = None,
    exponential_terms: Optional[Dict[str, List[int]]] = None,
    logarithmic_terms: Optional[Dict[str, List[int]]] = None,
) -> List[str]:
    """Get descriptive names for model coefficients with enhanced naming."""
    periods = periods or []
    step_dates = step_dates or []
    exponential_terms = exponential_terms or {}
    logarithmic_terms = logarithmic_terms or {}

    names = []

    # Polynomial terms with better names
    poly_names = [
        "constant",
        "linear_trend",
        "quadratic",
        "cubic",
        "quartic",
        "quintic",
    ]
    for i in range(polynomial_degree + 1):
        if i < len(poly_names):
            names.append(poly_names[i])
        else:
            names.append(f"polynomial_degree_{i}")

    # Periodic terms with descriptive names
    for period in periods:
        if abs(period - 1.0) < 1e-6:
            names.extend(["annual_sin", "annual_cos"])
        elif abs(period - 0.5) < 1e-6:
            names.extend(["semiannual_sin", "semiannual_cos"])
        elif abs(period - 1.0 / 3) < 1e-6:
            names.extend(["quarterly_sin", "quarterly_cos"])
        else:
            names.extend([f"period_{period:.3f}y_sin", f"period_{period:.3f}y_cos"])

    # Step function terms
    for step_date in step_dates:
        names.append(f"step_{step_date}")

    # Exponential decay terms
    for onset_date, char_times in exponential_terms.items():
        for tau_days in char_times:
            names.append(f"exp_decay_{onset_date}_tau{tau_days}d")

    # Logarithmic terms
    for onset_date, char_times in logarithmic_terms.items():
        for tau_days in char_times:
            names.append(f"log_term_{onset_date}_tau{tau_days}d")

    return names


def infer_parameters_from_names(coefficient_names: List[str]) -> Dict[str, Any]:
    """Infer model parameters from coefficient names.

    This is the inverse function of get_coefficient_names().

    Args:
        coefficient_names: List of coefficient names

    Returns:
        Dictionary with parameters that can be passed as kwargs:
        - polynomial_degree: int
        - periods: Optional[List[float]]
        - step_dates: Optional[List[str]]
        - exponential_terms: Optional[Dict[str, List[int]]]
        - logarithmic_terms: Optional[Dict[str, List[int]]]

    """
    polynomial_degree: int = 0
    periods: List[float] = []
    step_dates: List[str] = []
    exponential_terms: Dict[str, List[int]] = {}
    logarithmic_terms: Dict[str, List[int]] = {}

    # Mapping for polynomial terms
    poly_name_to_degree = {
        "constant": 0,
        "linear_trend": 1,
        "quadratic": 2,
        "cubic": 3,
        "quartic": 4,
        "quintic": 5,
    }

    for name in coefficient_names:
        # Check polynomial terms
        if name in poly_name_to_degree:
            polynomial_degree = max(polynomial_degree, poly_name_to_degree[name])
        elif name.startswith("polynomial_degree_"):
            degree = int(name.split("_")[-1])
            polynomial_degree = max(polynomial_degree, degree)

        # Check periodic terms
        elif name in ["annual_sin", "annual_cos"]:
            if 1.0 not in periods:
                periods.append(1.0)
        elif name in ["semiannual_sin", "semiannual_cos"]:
            if 0.5 not in periods:
                periods.append(0.5)
        elif name in ["quarterly_sin", "quarterly_cos"]:
            if 1.0 / 3 not in periods:
                periods.append(1.0 / 3)
        elif name.startswith("period_") and name.endswith(("y_sin", "y_cos")):
            # Extract period from pattern like 'period_{period:.3f}y_sin'
            match = re.match(r"period_(\d+\.\d+)y_(?:sin|cos)", name)
            if match:
                period = float(match.group(1))
                if period not in periods:
                    periods.append(period)

        # Check step function terms
        elif name.startswith("step_"):
            step_date = name[5:]  # Remove 'step_' prefix
            if step_date not in step_dates:
                step_dates.append(step_date)

        # Check exponential decay terms
        elif name.startswith("exp_decay_"):
            # Pattern: exp_decay_{onset_date}_tau{tau_days}d
            match = re.match(r"exp_decay_(.+)_tau(\d+)d", name)
            if match:
                onset_date = match.group(1)
                tau_days = int(match.group(2))
                if onset_date not in exponential_terms:
                    exponential_terms[onset_date] = []
                if tau_days not in exponential_terms[onset_date]:
                    exponential_terms[onset_date].append(tau_days)

        # Check logarithmic terms
        elif name.startswith("log_term_"):
            # Pattern: log_term_{onset_date}_tau{tau_days}d
            match = re.match(r"log_term_(.+)_tau(\d+)d", name)
            if match:
                onset_date = match.group(1)
                tau_days = int(match.group(2))
                if onset_date not in logarithmic_terms:
                    logarithmic_terms[onset_date] = []
                if tau_days not in logarithmic_terms[onset_date]:
                    logarithmic_terms[onset_date].append(tau_days)

    return {
        "polynomial_degree": polynomial_degree,
        "periods": periods or None,
        "step_dates": step_dates or None,
        "exponential_terms": exponential_terms or None,
        "logarithmic_terms": logarithmic_terms or None,
    }
