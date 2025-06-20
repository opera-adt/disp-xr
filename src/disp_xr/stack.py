import logging
from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .log.logging import log_runtime
from .product import _get_reference_dates

logger = logging.getLogger(__name__)

DEFAULT_CHUNKS = {"time": -1, "x": 512, "y": 512}


@log_runtime
def combine_disp_product(
    disp_df: pd.DataFrame,
    chunks: Optional[dict] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> xr.Dataset:
    """Stacks displacement products over time.

    Combines data from a DataFrame containing displacement file paths into an
    xarray dataset.

    Parameters
    ----------
    disp_df : pd.DataFrame
        DataFrame with displacement file paths and 'date1', 'date2' columns.
    chunks : Optional[dict], optional
        Chunking configuration for xarray. Defaults to None.
    start_date : Optional[str], optional
        Start date to filter data. Defaults to None.
    end_date : Optional[str], optional
        End date to filter data. Defaults to None.

    Returns
    -------
    xr.Dataset
        Stacked displacement dataset.

    """
    logger.info("Stacking into common stack")
    chunks = {
        **DEFAULT_CHUNKS,
        **(chunks or {}),
    }  # Merge default chunks with user-defined chunks
    logger.info(f" Chunk blocks: {chunks}")

    # Get substacks and reference dates
    substacks, reference_dates = _get_reference_dates(disp_df)

    mask = pd.Series(True, index=reference_dates)
    if start_date:
        mask &= pd.to_datetime(reference_dates) >= pd.to_datetime(start_date)

    if end_date:
        mask &= pd.to_datetime(reference_dates) <= pd.to_datetime(end_date)
    reference_dates = reference_dates[mask]

    stacks: list = []
    for ix, date in enumerate(reference_dates):
        stack_files = substacks.loc[date].sort_index().path.to_list()
        stack = xr.open_mfdataset(stack_files, chunks=chunks)

        # Append first epoch of new ministack to last epochs of previous
        if ix > 0:
            stack["displacement"] += stacks[ix - 1].isel(time=-1).displacement

        stacks.append(stack)

    # Get first reference date
    first_epoch = disp_df.start_date.min()
    first_epoch = np.datetime64(first_epoch.to_pydatetime())

    # Get empty dataset with first reference date
    first_ds = xr.full_like(stacks[0].isel(time=0), 0)
    first_ds["time"] = first_epoch
    first_ds["reference_time"] = first_ds["time"]
    first_ds = first_ds.expand_dims("time")

    # Concatenate first epoch with stacks
    stacks.insert(0, first_ds)

    return xr.concat(stacks, dim="time")


@log_runtime
def rebase_disp(
    disp_df: pd.DataFrame,
    chunks: Optional[dict] = None,
    add_reference_time: bool = False,
) -> xr.Dataset:
    """Stacks displacement products over time.

    Combines data from a DataFrame containing displacement file paths into an
    xarray dataset.

    Parameters
    ----------
    disp_df : pd.DataFrame
        DataFrame with displacement file paths and 'date1', 'date2' columns.
    chunks : Optional[dict], optional
        Chunking configuration for xarray. Defaults to None.
    start_date : Optional[str], optional
        Start date to filter data. Defaults to None.
    end_date : Optional[str], optional
        End date to filter data. Defaults to None.
    add_reference_time : bool, default False
        If True, adds 'reference_time' attribute to the output dataset,
        inferred from the earliest 'date1'.

    Returns
    -------
    xr.Dataset
        Stacked displacement dataset.

    """
    logger.info("Stacking into common stack")
    chunks = {
        **DEFAULT_CHUNKS,
        **(chunks or {}),
    }  # Merge default chunks with user-defined chunks
    logger.info(f" Chunk blocks: {chunks}")

    # Get substacks and reference dates
    sorted_paths = disp_df.sort_values("date2").path.to_list()

    ds_stack = xr.open_mfdataset(sorted_paths, chunks=chunks)

    # Get reference dates of ministacks and indices
    ref_dates = np.unique(ds_stack.reference_time)
    # Find the last date in each ministack
    last_ix = np.where(ds_stack.time.isin(ref_dates))[0]

    # Get offsets
    ref_offset = (ds_stack.displacement.isel(time=last_ix)).cumsum(dim="time")

    # Rebase displacement
    displacement_corrected = ds_stack.displacement.copy()

    for ix, ref in enumerate(ref_dates[1:]):
        mask = (ds_stack.reference_time == ref).compute()
        # Use xr.where to conditionally add the cumsum correction
        displacement_corrected = xr.where(
            mask,
            displacement_corrected + ref_offset.isel(time=ix),
            displacement_corrected,
        )

    # Update the dataset with corrected displacement
    ds_stack = ds_stack.assign(displacement=displacement_corrected)

    if add_reference_time:
        # Add initial reference epoch of zeros, and rechunk
        first_xr = xr.full_like(ds_stack.isel(time=0), 0)
        first_xr["time"] = ds_stack.reference_time[0]
        first_xr["reference_time"] = first_xr["time"]
        first_xr = first_xr.expand_dims("time")

        return xr.concat([first_xr, ds_stack], dim="time")
    else:
        return ds_stack


def _ensure_chunks(
    requested_chunks: dict[str, int] | None, data_shape: tuple[int, int, int]
) -> dict[str, int]:
    """Ensure requested_chunks are smaller than the downloaded size."""
    chunks = {**(requested_chunks or {})}
    chunks["time"] = min(chunks["time"], data_shape[0])
    chunks["y"] = min(chunks["y"], data_shape[1])
    chunks["x"] = min(chunks["x"], data_shape[2])
    return chunks


class NaNPolicy(str, Enum):
    """Policy for handling NaN values in rebase_timeseries."""

    propagate = "propagate"
    omit = "omit"

    def __str__(self) -> str:
        return self.value


# NOTE move bellow to function export to local zarr


def create_rebased_displacement(
    da_displacement: xr.DataArray,
    reference_datetimes: Sequence[datetime | pd.DatetimeIndex],
    process_chunk_size: tuple[int, int] = (512, 512),
    add_reference_time: bool = False,
    nan_policy: str | NaNPolicy = NaNPolicy.propagate,
) -> xr.DataArray:
    """Rebase and stack displacement products with different reference dates.

    This function combines displacement products that may have different reference
    dates by accumulating displacements when the reference date changes.
    When a new reference date is encountered, the displacement values from the
    previous stack's final epoch are added to all epochs in the new stack.

    Parameters
    ----------
    da_displacement : xr.DataArray
        Displacement dataarray to rebase.
    reference_datetimes : Sequence[datetime | pd.DatetimeIndex]
        Reference datetime for each epoch.
        Must be same length as `da_displacement.time`.
    process_chunk_size : tuple[int, int], optional
        Chunk size for processing. Defaults to (512, 512).
    add_reference_time : bool, optional
        Whether to add a zero array for the reference time.
        Defaults to False.
    nan_policy : choices = ["propagate", "omit"]
        Whether to propagate or omit (zero out) NaNs in the data.
        By default "propagate", which means any ministack, or any "reference crossover"
        product, with nan at a pixel causes all subsequent data to be nan.
        If "omit", then any nan causes the pixel to be zeroed out, which is
        equivalent to assuming that 0 displacement occurred during that time.

    Returns
    -------
    xr.DataArray
        Stacked displacement dataarray with rebased displacements.

    """
    logger.info("Starting displacement stack rebasing")

    process_chunks = {
        "time": -1,
        "y": process_chunk_size[0],
        "x": process_chunk_size[1],
    }
    process_chunks = _ensure_chunks(process_chunks, da_displacement.shape)

    # Make the map_blocks-compatible function to accumulate the displacement
    def process_block(arr: xr.DataArray) -> xr.DataArray:
        out = rebase_timeseries(
            arr.to_numpy(), reference_datetimes, nan_policy=nan_policy
        )
        return xr.DataArray(out, coords=arr.coords, dims=arr.dims)

    # Process the dataset in blocks
    rebased_da = da_displacement.chunk(process_chunks).map_blocks(process_block)

    if add_reference_time:
        # Add initial reference epoch of zeros, and rechunk
        rebased_da = xr.concat(
            [xr.full_like(rebased_da[0], 0), rebased_da],
            dim="time",
        )
        # Ensure correct dimension order
        rebased_da = rebased_da.transpose("time", "y", "x")

    return rebased_da


def rebase_timeseries(
    raw_data: np.ndarray,
    reference_dates: Sequence[datetime],
    nan_policy: str | NaNPolicy = NaNPolicy.propagate,
) -> np.ndarray:
    """Adjust for moving reference dates to create a continuous time series.

    DISP-S1 products have a reference date which changes over time.
    For example, shortening to YYYY-MM-DD notation, the products may be

        (2020-01-01, 2020-01-13)
        (2020-01-01, 2020-01-25)
        ...
        (2020-01-01, 2020-06-17)
        (2020-06-17, 2020-06-29)
        ...


    This function sums up the "crossover" values (the displacement image where the
    reference date moves forward) so that the output is referenced to the first input
    time.

    Parameters
    ----------
    raw_data : np.ndarray
        3D array of displacement values with moving reference dates
        shape = (time, rows, cols)
    reference_dates : Sequence[datetime]
        Reference dates for each time step
    nan_policy : choices = ["propagate", "omit"]
        Whether to propagate or omit (zero out) NaNs in the data.
        By default "propagate", which means any ministack, or any "reference crossover"
        product, with nan at a pixel causes all subsequent data to be nan.
        If "omit", then any nan causes the pixel to be zeroed out, which is
        equivalent to assuming that 0 displacement occurred during that time.

    Returns
    -------
    np.ndarray
        Continuous displacement time series with consistent reference date

    """
    if len(set(reference_dates)) == 1:
        return raw_data.copy()

    shape2d = raw_data.shape[1:]
    cumulative_offset = np.zeros(shape2d, dtype=np.float32)
    previous_displacement = np.zeros(shape2d, dtype=np.float32)

    # Set initial reference date
    current_reference_date = reference_dates[0]

    output = np.zeros_like(raw_data)
    # Process each time step
    for cur_ref_date, current_displacement, out_layer in zip(
        reference_dates, raw_data, output
    ):
        # Check for shift in temporal reference date
        if cur_ref_date != current_reference_date:
            # When reference date changes, accumulate the previous displacement
            if nan_policy == NaNPolicy.omit:
                np.nan_to_num(previous_displacement, copy=False)
            cumulative_offset += previous_displacement
            current_reference_date = cur_ref_date

        # Store current displacement for next iteration
        previous_displacement = current_displacement.copy()

        # Add cumulative offset to get consistent reference
        out_layer[:] = current_displacement + cumulative_offset

    return output
