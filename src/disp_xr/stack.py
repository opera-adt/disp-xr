import logging
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
