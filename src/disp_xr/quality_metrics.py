import logging
import warnings
from typing import Callable

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# Functions to run
def get_counts(array: np.ndarray, label: int = 0, reverse: bool = False) -> np.ndarray:
    """Count occurrences of a specific label in the array, with optional reverse.

    Parameters
    ----------
    array : np.ndarray
        The input array where occurrences of the label will be counted.
    label : int, optional
        The label to count. Default is 0.
    reverse : bool, optional
        Whether to count occurrences of labels not equal to the given label.
        Default is False.

    Returns
    -------
    np.ndarray
        The count of occurrences of the label (or non-labels if reverse is True).

    """
    kwargs = {"axis": 0, "keepdims": True}
    if reverse:
        return np.sum(array != label, **kwargs)
    else:
        return np.sum(array == label, **kwargs)


def func_stat(mode: str) -> Callable:
    """Return a statistical operation based on the specified mode.

    Parameters
    ----------
    mode : str
        The statistical operation to perform. Example values include 'mean',
        'median', 'max', 'min', 'std', 'var', and 'sum'.

    Returns
    -------
    callable
        A lambda function that performs the corresponding statistical operation
        on an xarray object.

    """
    kwargs = {"skipna": True, "keepdims": True}
    # Mapping mode to corresponding statistical operation
    operations = {
        "mean": lambda x: x.mean(dim="time", **kwargs),
        "median": lambda x: x.median(dim="time", **kwargs),
        "max": lambda x: x.max(dim="time", **kwargs),
        "min": lambda x: x.min(dim="time", **kwargs),
        "std": lambda x: x.std(dim="time", **kwargs),
        "var": lambda x: x.var(dim="time", **kwargs),
        "sum": lambda x: x.sum(dim="time", **kwargs),
    }

    # Check if the mode is valid and return the corresponding function
    try:
        return operations[mode]
    except KeyError:
        raise ValueError(
            (
                f"Unsupported mode: '{mode}'."
                " Valid options are 'mean', 'median', 'max',"
                " 'min', 'std', 'var', 'sum'"
            )
        )


def get_stack_stat(stack_xr: xr.DataArray, mode: str = "mean") -> xr.DataArray:
    """Compute a statistical operation (mean, median, etc.).

    Parameters
    ----------
    stack_xr : xr.DataArray
        The xarray DataArray on which the statistical operation..
    mode : str, optional
        The statistical operation to perform. Default is 'mean'.
        Options: 'median', 'max', 'min', 'std', 'var', and 'sum'.

    Returns
    -------
    xr.DataArray
        The result of the statistical operation, with the operation applied along the
        'time' dimension and any NaN values handled.

    """
    if not isinstance(stack_xr, xr.DataArray):
        raise TypeError("Input must be an xarray Dataset.")

    logger.info(f"Get {mode} of {stack_xr.name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        data = func_stat(mode)(stack_xr).compute()
    return data.squeeze()


# Map_block template
def get_template(xr_df: xr.DataArray) -> xr.DataArray:
    """Extract chunk sizes for y and x dimensions from xarray DataArray.

    Parameters
    ----------
    xr_df : xr.DataArray
        The xarray DataArray from which chunk sizes for the y and x dimensions
        will be extracted.

    Returns
    -------
    xr.DataArray
        A new xarray DataArray containing the chunk sizes.

    """
    # Get chunks (time, y, x)
    chunk_y = xr_df.chunks[1][0]  # use first layer
    chunk_x = xr_df.chunks[2][0]

    dim_time = len(xr_df.chunks[0])
    # Get shape
    shape = (dim_time, xr_df.sizes["y"], xr_df.sizes["x"])

    # Create an empty DataArray (NaNs by default)
    empty_data = np.full(shape, np.nan, dtype=np.float32)

    # Create xarray DataArray without 'time'
    empty_xarray = xr.DataArray(
        empty_data,
        dims=["time", "y", "x"],
        coords={"y": xr_df.y, "x": xr_df.x},
    )
    return empty_xarray.chunk(chunks={"time": 1, "y": chunk_y, "x": chunk_x})


def get_value_percentage(
    stack_xr: xr.DataArray, value: int, reverse: bool = False
) -> float:
    """Calculate the percentage of pixels in the xarray DataArray.

    Parameters
    ----------
    stack_xr : xr.DataArray
        The xarray DataArray to evaluate.
    value : int
        The value to check against in the DataArray.
    reverse : bool, optional
        Whether to calculate the percentage of pixels that do not match the value.
        Defaults to False (i.e., calculate percentage of pixels that match the value).

    Returns
    -------
    float
        The percentage of pixels in the DataArray that match (or do not) the value.

    """
    if not isinstance(stack_xr, xr.DataArray):
        raise TypeError("Input must be an xarray Dataset.")
    logger.info(f"Get percentage of {stack_xr.name}")
    template = get_template(stack_xr)
    data = stack_xr.map_blocks(
        get_counts, kwargs={"label": value, "reverse": reverse}, template=template
    )

    if template.sizes["time"] > 1:
        data = data.sum(dim="time").values
    else:
        data = np.squeeze(data.values)

    pct = data / np.int64(stack_xr.time.size) * 100
    return pct
