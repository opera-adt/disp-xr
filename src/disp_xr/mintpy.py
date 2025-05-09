import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from mintpy.utils import utils as ut
from mintpy.utils import writefile
from pyproj import CRS
from tqdm import tqdm

from .io import get_geospatial_info, open_image
from .product import get_disp_info
from .stack import combine_disp_product
from .utils import get_chunks_indices

logger = logging.getLogger(__name__)


def get_metadata(disp_nc: str | Path, reference_date: Optional[str] = None) -> dict:
    """Get metadata for MINTPY from a DISP NetCDF file.

    Args:
        disp_nc (str or Path): The path to the DISP NetCDF file.
        reference_date (str, optional): The reference date. Defaults to None.

    Returns:
        dict: A dictionary containing the metadata.

    """
    # Get high-level metadata from DISP
    ds = h5py.File(disp_nc, "r")
    length, width = ds["displacement"][:].shape

    # Get general metadata
    metadata = {}
    for key, value in ds.attrs.items():
        metadata[key] = value

    for key, value in ds["identification"].items():
        value = value[()]
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        metadata[key] = value

    for key, value in ds["metadata"].items():
        # Skip unnecessary keys
        if key not in ["reference_orbit", "secondary_orbit", "processing_information"]:
            metadata[key] = value[()]

    metadata["x"] = ds["x"][:]
    metadata["y"] = ds["y"][:]
    metadata["length"] = length
    metadata["width"] = width
    ds.close()
    del ds

    # Get geospatial information
    geo_info = get_geospatial_info(disp_nc)

    ## Prepare it in mintpy atr format
    metadata["LENGTH"] = geo_info.rows
    metadata["WIDTH"] = geo_info.cols

    metadata["X_FIRST"] = geo_info.gt[0]
    metadata["Y_FIRST"] = geo_info.gt[3]
    metadata["X_STEP"] = geo_info.gt[1]
    metadata["Y_STEP"] = geo_info.gt[5]
    metadata["GT"] = geo_info.transform
    metadata["X_UNIT"] = metadata["Y_UNIT"] = "meters"
    metadata["WAVELENGTH"] = metadata["radar_wavelength"]
    metadata["REF_DATE"] = reference_date

    # Projection and UTM zone
    proj = CRS.from_wkt(geo_info.crs.wkt)
    metadata["UTM_ZONE"] = proj.name.split(" ")[-1]
    metadata["EPSG"] = proj.to_authority()[-1]

    # Hardcoded values
    metadata["ALOOKS"] = metadata["RLOOkS"] = 1
    metadata["EARTH_RADIUS"] = 6371000.0  # Hardcoded
    metadata["FILE_TYPE"] = "timeseries"
    metadata["UNIT"] = "m"
    metadata["AZIMUTH_PIXEL_SIZE"] = 14.1  # where this comes from

    # Datetime
    t = pd.to_datetime(
        [
            metadata["reference_zero_doppler_start_time"],
            metadata["reference_zero_doppler_end_time"],
        ]
    )
    t_mid = t[0] + t.diff()[1] / 2
    total_seconds = (
        t_mid.hour * 3600 + t_mid.minute * 60 + t_mid.second + t_mid.microsecond / 1e6
    )
    metadata["CENTER_LINE_UTC"] = total_seconds

    # Clean up of metadata dicts
    for key in ["reference_datetime", "secondary_datetime"]:
        del metadata[key]

    return metadata


def write_geometry(
    disp_path: str | Path,
    reference_date: str,
    incidence_angle: str | Path,
    azimuth_angle: str | Path,
    dem_path: str | Path,
    water_mask_path: str | Path,
    output_dir: str | Path,
):
    """Generate and save a geometry file in HDF5 format.

    Parameters
    ----------
    disp_path : str or Path
        Path to the displacement file.
    reference_date : str
        Reference date for extracting metadata.
    incidence_angle : str or Path
        Path to the incidence angle file.
    azimuth_angle : str or Path
        Path to the azimuth angle file.
    dem_path : str or Path
        Path to the DEM (Digital Elevation Model) file.
    water_mask_path : str or Path
        Path to the water mask file.
    output_dir : str or Path
        Directory where the output geometry file will be saved.

    Returns
    -------
    None
        Writes the geometry data to "geometry.h5".

    Raises
    ------
    ValueError
        If any of the input files cannot be found.

    """
    # Create output directory if does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Open incidence and azimuth angles
    files = {}
    for txt, path in zip(
        ["inc_angle", "az_angle", "dem", "water_mask"],
        [incidence_angle, azimuth_angle, dem_path, water_mask_path],
    ):
        logger.info(f"Reading {path}")
        try:
            files[txt], _ = open_image(path)
        except FileNotFoundError:
            raise ValueError(f"File {path} not found")

    # Get metadata
    # NOTE suspend temporary logging here
    metadata = get_metadata(disp_path, reference_date=reference_date)

    # Save mintpy geometry
    logger.info(f"Writing geometry.h5 to {output_dir}")
    meta = dict(metadata.items())
    meta["FILE_TYPE"] = "geometry"

    dsDict = {
        "incidenceAngle": files["inc_angle"],
        "azimuthAngle": files["az_angle"],
        "height": files["dem"],
        "waterMask": files["water_mask"],
    }

    writefile.write(dsDict, Path(output_dir) / "geometry.h5", metadata=meta)


## TIMESERIES
def process_and_write(
    i: Tuple[int, ...],
    f: h5py.File,
    stack,  # Assuming `Stack` is a custom class, specify its module if needed
    ref_data: np.ndarray,
    mask: bool = True,
    drop_nan: bool = True,
) -> None:
    """Process displacement data and write it to an HDF5 file.

    Parameters
    ----------
    i : tuple of int
        The index of the data slice to be processed and written.
    f : h5py.File
        The HDF5 file object where processed data will be written.
    stack : Stack
        The stack object containing the displacement data.
    ref_data : np.ndarray
        The reference data to be subtracted from the displacement data.
    mask : bool, optional
        Whether to apply a mask to the data. Defaults to True.
    drop_nan : bool, optional
        Whether to convert NaN values to 0 and apply a water mask. Defaults to True.

    Returns
    -------
    None
        The function processes the data and writes it to HDF5 file.

    Raises
    ------
    ValueError
        If input data dimensions do not match expectations.

    """
    # Create a mask based on the unwrapper mask and apply it to the data
    subset = stack.isel(time=i[0], y=i[1], x=i[2]).compute()
    data = subset.displacement.values

    # Change slice to skip writing to the first data
    write_ix = (slice(1, None, None), i[1], i[2])

    # Use mask
    if mask:
        mask = subset.recommended_mask.values == 0
    else:
        mask = subset.water_mask.values == 0

    data = np.ma.masked_array(data, mask=mask).filled(np.nan)

    # Re-reference: subtract reference data
    if ref_data is not None:
        if data.ndim == 3:
            ref_tile = np.tile(
                ref_data.reshape(-1, 1, 1), (1, data.shape[-2], data.shape[-1])
            )
        else:
            ref_tile = np.tile(
                ref_data.reshape(-1, 1), (data.shape[-2], data.shape[-1])
            )
        data -= ref_tile

    # Convert NaNs to 0 and apply water mask
    if drop_nan:
        data = np.nan_to_num(data)

    # Write the re-referenced data to the "timeseries" dataset
    f["timeseries"][write_ix] = data

    # Add connected component labels to the "connectComponent" dataset
    f["connectComponent"][write_ix] = subset.connected_component_labels.values


def write_data_parallel(
    output: Union[str, h5py.File],
    stack: np.ndarray,
    ref_data: np.ndarray,
    chunks_ix: List[tuple],
    mask: bool = True,
    drop_nan: bool = True,
    num_threads: int = 4,
) -> None:
    """Write data to an HDF5 file in append mode using parallel execution.

    Parameters
    ----------
    output : str or h5py.File
        The path to the HDF5 file or an open HDF5 file object.
    stack : np.ndarray
        The stack of displacement data to be written.
    ref_data : np.ndarray
        The reference data used for processing.
    chunks_ix : list of tuple
        List of chunk indices defining portions of the data to process.
    mask : bool, optional
        Whether to apply a mask to the data. Defaults to True.
    drop_nan : bool, optional
        Whether to drop NaN values from the data. Defaults to True.
    num_threads : int, optional
        The number of threads to use for parallel execution. Defaults to 4.

    Returns
    -------
    None
        The function writes processed data to the HDF5 file.

    Raises
    ------
    ValueError
        If the HDF5 file path is invalid.

    """
    logger.info("Writing data to HDF5 file in append mode...")

    # Initialize the HDF5 file and the progress bar
    with h5py.File(output, "a") as f:

        def process_and_write2(x):
            return process_and_write(
                x, f, stack, ref_data, mask=mask, drop_nan=drop_nan
            )

        # Use ThreadPoolExecutor for parallel execution
        # TODO replace this with dask futures
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(
                tqdm(executor.map(process_and_write2, chunks_ix), total=len(chunks_ix))
            )

    logger.info("Data writing complete.")


def write_mintpy_container_parallel(
    stack_xr: xr.Dataset,
    metadata: Dict[str, Union[str, float, int, list]],
    ref_lalo: List[float],
    output: Union[Path, str] = "timeseries.h5",
    mask: bool = True,
    drop_nan: bool = True,
    num_threads: int = 4,
) -> None:
    """Write MintPy container in parallel.

    Parameters
    ----------
    stack_xr : xr.Dataset
        The stack of data to be written.
    metadata : dict
        The metadata defining the MintPy layout.
    ref_lalo : list of float
        The reference point latitude and longitude for the stack.
    output : str, optional
        Path to the output HDF5 file. Defaults to 'timeseries.h5'.
    mask : bool, optional
        Whether to apply a mask to the data. Defaults to True.
    drop_nan : bool, optional
        Whether to drop NaN values from the data. Defaults to True.
    num_threads : int, optional
        The number of threads for parallel execution. Defaults to 4.

    Returns
    -------
    None
        The function writes the MintPy container to an HDF5 file.

    Raises
    ------
    ValueError
        If the dataset is empty or incorrectly formatted.

    """
    # map chunks indices
    chunks_ix = get_chunks_indices(stack_xr)
    logger.info(f"number of chunks: {len(chunks_ix)}")
    stack_xr["displacement"].attrs["units"] = "m"

    # Get metadata for MintPy layout
    date_list = list(stack_xr["time"].dt.strftime("%Y%m%d").data)
    date_list = [metadata["REF_DATE"]] + date_list
    dates = np.array(date_list, dtype=np.bytes_)
    num_date = len(date_list)
    rows, cols = stack_xr.displacement.shape[1:]

    # pbase
    pbase = np.zeros(num_date, dtype=np.float32)

    # define dataset structure
    dates = np.array(date_list, dtype=np.bytes_)

    ds_name_dict = {
        "date": [dates.dtype, (num_date,), dates],
        "bperp": [np.float32, (num_date,), pbase],
        "timeseries": [np.float32, (num_date, rows, cols), None],
        "connectComponent": [np.int16, (num_date, rows, cols)],
    }

    # Change spatial reference
    if ref_lalo is not None:
        coord = ut.coordinate(metadata)
        ref_lat, ref_lon = ut.latlon2utm(metadata, *ref_lalo)
        yx = coord.lalo2yx(ref_lat, ref_lon)
        logger.info(f"Re-referencing to x:{yx[1]}, y:{yx[0]}")
        metadata["REF_LAT"] = ref_lat
        metadata["REF_LON"] = ref_lon
        metadata["REF_Y"] = yx[0]
        metadata["REF_X"] = yx[1]

        # Get reference data
        ref_data = stack_xr.displacement.isel(x=yx[1], y=yx[0]).values
    else:
        ref_data = None

    # initiate HDF5 file
    writefile.layout_hdf5(output, ds_name_dict, metadata=metadata)

    # Write
    write_data_parallel(
        output,
        stack_xr,
        ref_data,
        chunks_ix,
        mask=mask,
        drop_nan=drop_nan,
        num_threads=num_threads,
    )


def create_timeseries_h5(
    products_path: str,
    output_dir: str | Path,
    ref_lalo: Optional[List[float]] = None,
    ref_yx: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    mask: bool = True,
    drop_nan: bool = True,
    num_threads: int = 5,
) -> None:
    """Create an HDF5 time series file from displacement products.

    Parameters
    ----------
    products_path : str
        Path to the directory containing displacement product files.
    output_dir : str
        Path where the output HDF5 file will be saved.
    ref_lalo : list of float, optional
        Reference latitude and longitude [lat, lon]. Defaults to None.
    ref_yx : list of int, optional
        Reference pixel coordinates [y, x] in the image grid. Defaults to None.
    start_date : str, optional
        Start date of the time series (YYYY-MM-DD). Defaults to None.
    mask : bool, optional
        Whether to apply a mask to filter out invalid data. Defaults to True.
    drop_nan : bool, optional
        Whether to replace NaN values with 0. Defaults to True.
    num_threads : int, optional
        Number of parallel threads to use. Defaults to 5.

    Raises
    ------
    FileNotFoundError
        If `products_path` does not exist.
    ValueError
        If `output_dir` cannot be created.

    """
    logger.info("Starting time series HDF5 creation...")

    # Start dask client to speed up reading
    client = Client(n_workers=num_threads)
    logger.info(f"Dask: {client.dashboard_link}")
    try:
        # Get filename information from the DISP products
        disp_df = get_disp_info(products_path)

        output_dir = Path(output_dir).absolute()
        output_dir.mkdir(exist_ok=True, parents=True)

        # Load stacks
        stack = combine_disp_product(disp_df, start_date=start_date)

        # Combine along time
        stack = stack.chunk({"time": -1})
        stack = stack.transpose("time", "y", "x")

        # Get metadata from the DISP NetCDF file
        ref_date = pd.to_datetime(np.min(stack.reference_time.values))
        logger.info(f"Reference date: {ref_date}")
        meta = get_metadata(disp_df.path.iloc[0], ref_date.strftime("%Y%m%d"))

        # Select only variables needed
        small_stack = stack[
            [
                "displacement",
                "water_mask",
                "recommended_mask",
                "connected_component_labels",
            ]
        ]

        # Get reference point
        if (ref_lalo is None) & (ref_yx is not None):
            ref_utm = ut.coordinate(meta).yx2lalo(*ref_yx)
            ref_lalo = list(ut.utm2latlon(meta, *reversed(ref_utm)))
        else:
            raise ValueError("Specify either ref_lalo, or ref_yx")
        logger.info(f"Reference point: lat: {ref_lalo[0]:.2f}, lon: {ref_lalo[1]:.2f}")
        # Get this size of one chunk
        chunk_mb = small_stack.displacement[get_chunks_indices(stack)[0]].nbytes / (
            1024**2
        )
        logger.info(f"Chunk size: {chunk_mb} MB")
        logger.info(f"Size of parallel run: {chunk_mb * num_threads} MB")

        # Call the write_mintpy_container_parallel function
        write_mintpy_container_parallel(
            small_stack,
            meta,
            ref_lalo=ref_lalo,
            output=output_dir / "timeseries.h5",
            mask=mask,
            drop_nan=drop_nan,
            num_threads=num_threads,
        )
        logger.info(f'Output file: {output_dir / "timeseries.h5"}')
        client.close()
        logger.info("Dask client closed.")
    except Exception as e:
        client.close()
        logger.info("Dask client closed.")
        print(e)
