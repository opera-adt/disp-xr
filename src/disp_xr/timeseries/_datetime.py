from datetime import datetime, timezone

import numpy as np


def _dates_to_decimal_years(dates: np.ndarray, ref_idx: int) -> np.ndarray:
    """Convert dates to decimal years relative to reference date.

    Parameters
    ----------
    dates : np.ndarray
        Array of dates (datetime64 or datetime objects)
    ref_idx : int
        Index of reference date

    Returns
    -------
    np.ndarray
        Decimal years relative to reference date

    """
    if len(dates) > 0 and isinstance(dates[0], np.datetime64):
        # Handle numpy datetime64 arrays efficiently
        decimal_years = np.array([_datetime64_to_decimal_year(date) for date in dates])
    else:
        # Handle datetime objects
        decimal_years = np.array([_datetime_to_decimal_year(date) for date in dates])

    return decimal_years - decimal_years[ref_idx]


def _datetime64_to_decimal_year(date: np.datetime64) -> float:
    """Convert numpy datetime64 to decimal year."""
    try:
        ts = date.astype("datetime64[s]").astype(int)
        dt = datetime.fromtimestamp(ts, timezone.utc)
        return _datetime_to_decimal_year(dt)
    except (ValueError, OverflowError):
        # Handle potential overflow for extreme dates
        return float("nan")


def _datetime_to_decimal_year(dt: datetime) -> float:
    """Convert datetime object to decimal year.

    Uses more accurate calculation that accounts for leap years.
    """
    year_start = datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)
    year_end = datetime(dt.year + 1, 1, 1, tzinfo=dt.tzinfo)
    year_length = (year_end - year_start).total_seconds()
    elapsed = (dt - year_start).total_seconds()

    return dt.year + elapsed / year_length


def _date_str_to_decimal_year(date_str: str) -> float:
    """Convert date string (YYYYMMDD format) to decimal year.

    Parameters
    ----------
    date_str : str
        Date string in YYYYMMDD format

    Returns
    -------
    float
        Decimal year representation

    """
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
        return _datetime_to_decimal_year(dt)
    except ValueError:
        raise ValueError(f"Invalid date string '{date_str}'. Expected YYYYMMDD format")
