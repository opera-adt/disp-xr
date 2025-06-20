import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def get_disp_info(
    products_path: Path | str, min_version: Optional[str] = None
) -> pd.DataFrame:
    """Get filename information from OPERA DISP products.

    Parameters
    ----------
    products_path : Union[Path, str]
        The path to the OPERA DISP products.
    min_version : Optional[str], optional
        Minimum version to filter the products. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the filename information.

    """
    # Get all OPERA DISP products in the specified path
    disp_products = list(Path(products_path).rglob("*.nc"))
    logger.info(f"Found OPERA DISP: {len(disp_products)} products")

    if not disp_products:
        return pd.DataFrame()

    # Parse filenames in batch - more efficient than list comprehension
    filename_parts = [product.stem.split("_") for product in disp_products]

    # Create DataFrame with all data at once
    df = pd.DataFrame(
        filename_parts,
        columns=[
            "project",
            "level",
            "product",
            "mode",
            "frame_id",
            "polarization",
            "start_date",
            "end_date",
            "version",
            "production_date",
        ],
    )

    # Add path column
    df["path"] = disp_products
    start_dates = df["start_date"].str.split("T", expand=True)[0]
    end_dates = df["end_date"].str.split("T", expand=True)[0]

    df["date12"] = start_dates + "_" + end_dates
    df["date1"] = start_dates
    df["date2"] = end_dates

    # Convert dates to datetime
    date_format = "%Y%m%dT%H%M%SZ"
    df["start_date"] = pd.to_datetime(df["start_date"], format=date_format)
    df["end_date"] = pd.to_datetime(df["end_date"], format=date_format)

    logger.info(f" Starting date: {df.start_date.min()}")
    logger.info(f" Ending date: {df.end_date.max()}")

    # Filter version if multiple versions are present
    unique_versions = df.version.unique()
    if len(unique_versions) > 1:
        logger.info(f" Versions: {unique_versions}")
        if min_version is None:
            min_version = unique_versions.max()
        df = df[df["version"] == min_version].copy()
        logger.info(f" filtered with {min_version}: {len(df)} products")

    # Get number of reference dates
    logger.info(f" Number of reference dates: {len(_get_reference_dates(df)[1])}")

    # Fixed the return statement (assuming find_duplicates returns tuple)
    return _find_duplicates(df)[0].sort_values(by="date12", ignore_index=True)


def _get_reference_dates(df: pd.DataFrame) -> Union[pd.DataFrame, List]:
    substacks = df.groupby(["date1", "date2"]).apply(lambda x: x, include_groups=False)
    reference_dates = substacks.index.get_level_values(0).unique()
    return substacks, reference_dates


def _find_duplicates(input_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find and remove duplicates in the input DataFrame based on the 'date12' column.

    Parameters
    ----------
    input_df : pandas.DataFrame
        The input DataFrame containing the data.

    Returns
    -------
    tuple
        A tuple containing two DataFrames. The first DataFrame is the input DataFrame
        with duplicates removed, and the second DataFrame contains the removed
        duplicate rows.

    """
    input_df2 = input_df.copy()
    list_duplicates = input_df2.date12.value_counts()
    duplicates = list_duplicates[list_duplicates.values > 1].index

    duplicate_list = []
    for date in duplicates:
        selected_df = input_df2[input_df2.date12 == date]
        latest_production_date = selected_df.production_date.max()
        for ix, key in (selected_df.production_date != latest_production_date).items():
            if key is True:
                input_df2.drop(ix, inplace=True)
            else:
                duplicate_list.append(input_df.iloc[ix])

    if len(duplicate_list) > 0:
        duplicate_list = pd.concat(duplicate_list, axis=1).T

    logger.info(f" Skip {len(duplicate_list)} duplicates")

    return input_df2, duplicate_list
