import numpy as np
import xarray as xr
from scipy import stats


def lag_autocorr(dxr: xr.DataArray, lag: int = 1) -> xr.DataArray:
    """Lag-N autocorrelation."""
    n_valid = dxr.count(dim="time")

    # Mean and center
    x_mean = dxr.mean(dim="time", skipna=True)
    x_centered = dxr - x_mean

    x_t = x_centered.isel(time=slice(0, -lag))
    x_t_lag = x_centered.isel(time=slice(lag, None))

    # Numerator
    numerator = np.nansum(x_t.data * x_t_lag.data, axis=0)
    denominator = (x_centered**2).sum(dim="time", skipna=True)

    # Division
    autocorr = xr.where(
        (denominator > 1e-15) & (n_valid >= lag + 2), numerator / denominator, np.nan
    )

    return autocorr


def compute_moments(data, mean_val, std_val):
    """Compute 3rd and 4th moments efficiently."""
    # Centered data
    centered = data - mean_val

    # Compute powers sequentially to minimize memory
    moment3 = (centered**3).mean(dim="time", skipna=True)
    moment4 = (centered**4).mean(dim="time", skipna=True)

    # Standardized moments
    skew = xr.where(std_val > 0, moment3 / (std_val**3), 0)
    kurt = xr.where(std_val > 0, moment4 / (std_val**4) - 3, 0)

    return skew, kurt


def anderson_darling(data, min_samples=20):
    """Calculate Anderson-Darling test statistic for a single pixel time series."""
    data_clean = data[~np.isnan(data)]  # remove NaNs if any
    if len(data_clean) > min_samples:  # require minimum number of samples to test
        result = stats.anderson(data_clean, dist="norm")
        return result.statistic
    else:
        return np.nan


# NOte see if I can optimaze memory
def get_residuals_analysis(residuals: xr.DataArray) -> xr.Dataset:
    kwargs = {"skipna": True}
    mean_residual = residuals.mean(dim="time", **kwargs)
    std_residual = residuals.std(dim="time", **kwargs)

    # Skewness and kurtosis
    skewness, kurtosis = compute_moments(residuals, mean_residual, std_residual)

    # autocorrelation
    autocorr = lag_autocorr(residuals)

    return xr.Dataset(
        {
            "residual_mean": mean_residual,
            "residual_std": std_residual,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "autocorr": autocorr,
        }
    )


# This one is problematic
def residual_normality_test(residuals: xr.DataArray) -> xr.DataArray:
    # normal distribution test
    ufunc_kwargs = {
        "input_core_dims": [["time"]],
        "output_core_dims": [[]],
        "vectorize": True,
        "dask": "parallelized",
        "dask_gufunc_kwargs": {"allow_rechunk": True},
        "output_dtypes": [float],
        "keep_attrs": True,
    }
    return xr.apply_ufunc(anderson_darling, residuals, **ufunc_kwargs)
