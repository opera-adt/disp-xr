from __future__ import annotations

import logging
import re
import warnings
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import requests

__all__ = ["search"]

logger = logging.getLogger("opera_utils")

DISP_FILE_REGEX = re.compile(
    "OPERA_L3_DISP-"
    r"(?P<sensor>(S1|NI))_"
    r"(?P<acquisition_mode>IW)_"  # TODO: What's NISAR's?
    r"F(?P<frame_id>\d{5})_"
    r"(?P<polarization>(VV|HH))_"
    r"(?P<reference_datetime>\d{8}T\d{6}Z)_"
    r"(?P<secondary_datetime>\d{8}T\d{6}Z)_"
    r"v(?P<version>[\d.]+)_"
    r"(?P<generation_datetime>\d{8}T\d{6}Z)",
)


class UrlType(str, Enum):
    """Choices for the orbit direction of a granule."""

    S3 = "s3"
    HTTPS = "https"

    def __str__(self) -> str:
        return str(self.value)


def from_filename(name: Path | str):
    """Parse a filename to create a DispProduct.

    Parameters
    ----------
    name : str or Path
        Filename to parse for OPERA DISP-S1 information.

    Returns
    -------
    DispProduct
        Parsed file information.

    Raises
    ------
    ValueError
        If the filename format is invalid.

    """

    def _to_datetime(dt: str) -> datetime:
        return datetime.strptime(dt, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)

    if not (match := DISP_FILE_REGEX.match(Path(name).name)):
        raise ValueError(f"Invalid filename format: {name}")

    data: dict[str, Any] = match.groupdict()
    data["reference_datetime"] = _to_datetime(data["reference_datetime"])
    data["secondary_datetime"] = _to_datetime(data["secondary_datetime"])
    data["generation_datetime"] = _to_datetime(data["generation_datetime"])
    data["frame_id"] = int(data["frame_id"])
    data["filename"] = name
    return data


def _get_download_url(
    umm_data: dict[str, Any], protocol: UrlType = UrlType.HTTPS
) -> str:
    """Extract a download URL from the product's UMM metadata.

    Parameters
    ----------
    umm_data : dict[str, Any]
        The product's umm metadata dictionary
    protocol : UrlType
        The protocol to use for downloading, either "s3" or "https"

    Returns
    -------
    str
        The download URL

    Raises
    ------
    ValueError
        If no URL with the specified protocol is found or if the protocol is invalid

    """
    if protocol not in ["https", "s3"]:
        raise ValueError(f"Unknown protocol {protocol}; must be https or s3")

    for url in umm_data["RelatedUrls"]:
        if url["Type"].startswith("GET DATA") and url["URL"].startswith(protocol):
            return url["URL"]

    raise ValueError(f"No download URL found for granule {umm_data['GranuleUR']}")


def from_umm(umm_data: dict[str, Any], url_type: UrlType = UrlType.HTTPS):
    """Construct a DispProduct instance from a raw dictionary.

    Parameters
    ----------
    umm_data : dict[str, Any]
        The raw granule UMM data from the CMR API.
    url_type : UrlType
        Type of url to use from the Product.
        "s3" for S3 URLs (direct access), "https" for HTTPS URLs.

    Returns
    -------
    Granule
        The parsed Granule instance.

    Raises
    ------
    ValueError
        If required temporal extent data is missing.

    """
    url = _get_download_url(umm_data, protocol=url_type)
    product_dict = from_filename(url)

    archive_info = umm_data.get("DataGranule", {}).get(
        "ArchiveAndDistributionInformation", []
    )
    size_in_bytes = archive_info[0].get("SizeInBytes", 0) if archive_info else None
    product_dict["size_in_bytes"] = size_in_bytes
    return product_dict


def search(
    frame_id: int | None = None,
    product_version: str | None = "1.0",
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    url_type: UrlType = UrlType.HTTPS,
    use_uat: bool = False,
) -> pd.DataFrame:
    """Query the CMR for granules matching the given frame ID and product version.

    Parameters
    ----------
    frame_id : int, optional
        The frame ID to search for
    product_version : str, optional
        The product version to search for
    start_datetime : datetime, optional
        The start of the temporal range in UTC.
    end_datetime : datetime, optional
        The end of the temporal range in UTC.
    url_type : UrlType
        The protocol to use for downloading, either "s3" or "https".
    use_uat : bool
        Whether to use the UAT environment instead of main Earthdata endpoint.

    Returns
    -------
    list[DispProduct]
        List of products matching the search criteria

    """
    edl_host = "uat.earthdata" if use_uat else "earthdata"
    search_url = f"https://cmr.{edl_host}.nasa.gov/search/granules.umm_json"
    params: dict[str, int | str | list[str]] = {
        "short_name": "OPERA_L3_DISP-S1_V1",
        "page_size": 2000,
    }
    # Optionally narrow search by frame id, product version
    product_filters: list[str] = []
    if product_version:
        product_filters.append(f"float,PRODUCT_VERSION,{product_version}")
    if product_filters:
        params["attribute[]"] = product_filters

    # Optionally narrow search by temporal range
    if start_datetime is not None or end_datetime is not None:
        start_str = start_datetime.isoformat() if start_datetime is not None else ""
        end_str = end_datetime.isoformat() if end_datetime is not None else ""
        params["temporal"] = f"{start_str},{end_str}"

    # If no temporal range is specified, default to all granules
    # Ensure datetime objects are timezone-aware
    if start_datetime is None:
        start_datetime = datetime(2014, 1, 1, tzinfo=timezone.utc)
    else:
        start_datetime = start_datetime.replace(tzinfo=timezone.utc)
    if end_datetime is None:
        end_datetime = datetime(2100, 1, 1, tzinfo=timezone.utc)
    else:
        end_datetime = end_datetime.replace(tzinfo=timezone.utc)

    if frame_id:
        product_filters.append(f"int,FRAME_NUMBER,{frame_id}")
    else:
        warnings.warn("No `frame_id` specified: search may be large", stacklevel=1)

    headers: dict[str, str] = {}
    products: list = []
    while True:
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        # Get url and data
        product_dict = [from_umm(ix["umm"], url_type=url_type) for ix in data["items"]]
        products.append(pd.DataFrame(product_dict))

        if "CMR-Search-After" not in response.headers:
            break

        headers["CMR-Search-After"] = response.headers["CMR-Search-After"]

    products_df = pd.concat(products, ignore_index=True)

    mask = (products_df.secondary_datetime >= start_datetime) & (
        products_df.secondary_datetime <= end_datetime
    )

    # Return sorted list of products
    return products_df[mask]
