import logging
import random
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import xarray as xr
import yaml  # type: ignore
from pyproj import Transformer

logger = logging.getLogger(__name__)


def get_chunks_indices(xr_array: xr.Dataset) -> list:
    """Get the indices for chunked slices of an xarray Dataset.

    Determines the chunk boundaries of the given xarray and returns a list
    of slice objects representing these chunks.

    Parameters
    ----------
    xr_array : xr.Dataset
        The input xarray Dataset.

    Returns
    -------
    list
        A list of slice objects representing the chunked slices.

    """
    chunks = xr_array.chunks
    _, iy, ix = chunks["time"], chunks["y"], chunks["x"]

    idx = [sum(ix[:i]) for i in range(len(ix) + 1)]
    idy = [sum(iy[:i]) for i in range(len(iy) + 1)]

    slices = []

    for i in range(len(idy) - 1):  # Y-axis slices for idy
        for j in range(len(idx) - 1):  # X-axis slices for idx
            # Create a slice using the ranges of idt, idy, and idx
            # skip first date
            slice_ = np.s_[:, idy[i] : idy[i + 1], idx[j] : idx[j + 1]]
            slices.append(slice_)
    return slices


def latlon_to_utm(lat: float, lon: float, utm_epsg: int) -> Tuple[float, float]:
    """Convert latitude and longitude coordinates to UTM coordinates.

    Parameters
    ----------
    lat : float
        Latitude coordinate in decimal degrees.
    lon : float
        Longitude coordinate in decimal degrees.
    utm_epsg : int
        The EPSG code for the UTM zone to which the coordinates should be transformed.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the UTM easting and northing coordinates.

    """
    transformer = Transformer.from_crs("epsg:4326", f"epsg:{utm_epsg}", always_xy=True)
    return transformer.transform(lon, lat)


def get_extent(ds: xr.Dataset) -> Tuple[float, float, float, float]:
    """Calculate the DISP spatial extent in UTM.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset, which must have a spatial reference
        with a 'GeoTransform' attribute and 'x' and 'y' dimensions.

    Returns
    -------
    Tuple[float, float, float, float]
        A tuple representing the extent (x0, x1, y1, y0) where:
        - x0 is the starting x-coordinate,
        - x1 is the ending x-coordinate,
        - y1 is the starting y-coordinate,
        - y0 is the ending y-coordinate.

    """
    # Extracting GeoTransform values
    transform = ds.spatial_ref.attrs["GeoTransform"].split(" ")
    x0, dx, _, y0, _, dy = map(float, transform)

    # Calculating the extent
    x1 = x0 + dx * ds.sizes["x"]
    y1 = y0 + dy * ds.sizes["y"]

    return (x0, x1, y1, y0)


def get_burst_ids(disp_path: Union[str, Path]) -> tuple[Optional[str], Optional[str]]:
    """Extract burst ID and track ID from a DISP file path.

    Parameters
    ----------
    disp_path : str | Path
        Path to the DISP file.

    Returns
    -------
    tuple[Optional[str], Optional[str]]
        A tuple containing the burst ID and track ID. If not found, returns None.

    """

    def _get_id(path: str) -> Optional[str]:
        """Extract the burst ID from the file name."""
        match = re.search(r"\d{6}-IW\d", path.split("/")[-1])
        if match:  # Check if a match was found
            return match.group(0)
        return None  # Return None if no match is found

    def _get_track(path: str) -> Optional[str]:
        """Extract the track ID from the file name."""
        match = re.search(r"T\d{3}", path.split("/")[-1])
        if match:  # Check if a match was found
            return match.group(0)
        return None  # Return None if no match is found

    # Get static layers from metadata
    with h5py.File(disp_path, "r") as f:
        runconfig = f["metadata"]["pge_runconfig"][()].decode()
        runconfig = yaml.safe_load(runconfig)
        static_lyrs = runconfig["dynamic_ancillary_file_group"]
        static_lyrs = static_lyrs["static_layers_files"]

    # Get burst and track ids from static layers
    burst_ids = np.unique(list(map(_get_id, static_lyrs)))
    track_id = np.unique(list(map(_get_track, static_lyrs)))[0]

    # Reformat to asf_vertex format
    burst_ids = [track_id + "_" + burst for burst in burst_ids]
    burst_ids = [re.sub(r"-", "_", burst) for burst in burst_ids]
    return burst_ids


def find_reference_point(
    pct_mask: np.ndarray,
    pct_ps: np.ndarray,
    phase_jumps: Optional[np.ndarray] = None,
    percentile: int = 99,
    max_jumps: int = 1,
) -> Tuple[int, int]:
    """Select a reference point based on PS stability and mask constraints.

    The function identifies a reference point where the PS (Persistent Scatterer)
    percentage is above the specified percentile and the percentage mask is nonzero.
    If phase jump data is provided, it is used to further refine the selection by
    excluding PS pixels with excessive phase jumps.

    Parameters
    ----------
    pct_mask : np.ndarray
        Percentage mask where nonzero values indicate valid regions.
    pct_ps : np.ndarray
        Percentage of PS stability in the stack (100% remains PS).
    phase_jumps : Optional[np.ndarray], optional
        Number of 2π phase jumps in the stack. Used for refinement. Defaults to None.
    percentile : int, optional
        Percentile threshold for PS stability selection. Defaults to 99.
    max_jumps : int, optional
        Maximum allowable phase jumps for valid reference points. Defaults to 1.

    Returns
    -------
    Tuple[int, int]
        The (y, x) coordinates of the selected reference point.

    """
    # Get valid pixes by masking out 100% recommended mask and 0% of pixels in the stack
    valid_pixels = np.ma.masked_where((pct_ps == 0) | (pct_mask == 100), pct_ps)

    # Compute 90th percentile threshold over valid pixels
    threshold = np.nanpercentile(valid_pixels.filled(np.nan), percentile)

    # Create a mask for high-coherence pixels
    valid = valid_pixels >= threshold

    # if timeseries inversion residualt metric is specified
    # remove pixels with > 1 phase jumps
    if phase_jumps is not None:
        valid[np.where((valid == 1) & (phase_jumps > max_jumps))] = 0

    # Get indices of valid pixels
    valid_indices = np.argwhere(valid.filled(0))

    if valid_indices.size == 0:
        raise ValueError(
            ("No valid reference point found.Adjust threshold or check input data.")
        )

    # Randomly select one valid pixel
    y, x = np.squeeze(random.choice(valid_indices))

    # Flip y-axis if needed (assuming `y` is inverted)
    # y = mean_tcoh.shape[0] - 1 - y

    logger.info(f"Selected reference pixel (y/x): {(y, x)}")

    return y, x
