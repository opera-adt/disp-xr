import logging
import shutil
import warnings
from pathlib import Path
from typing import Union

import asf_search as asf
import dem_stitcher
import numpy as np
import tile_mate
from opera_utils.geometry import stitch_geometry_layers
from pyproj import Transformer
from rasterio.warp import Resampling

from .io import get_geospatial_info, open_image, reproject_raster, write_geotiff
from .utils import get_burst_ids

logger = logging.getLogger(__name__)


def get_static_layers(
    disp_path: Union[str, Path], output_dir: Union[str, Path], n_workers: int = 5
) -> dict:
    """Extract static layers for DISP data.

    Processes the DISP dataset to generate static layers and saves them
    to the specified output directory.

    Parameters
    ----------
    disp_path : str | Path
        Path to the DISP dataset.
    output_dir : str | Path
        Directory where the output static layers will be saved.
    n_workers : int, optional
        Number of parallel workers to use. Defaults to 5.

    Returns
    -------
    dict
        Dictionary containing metadata or paths of the extracted static layers.

    """
    # Create output directory if does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create temporary directory for downloading static layers
    temp_dir = output_dir / "tmp"
    temp_dir.mkdir(exist_ok=True)

    # Get burst ids
    burst_ids = get_burst_ids(disp_path)

    # Download static layers
    results = asf.search(
        operaBurstID=list(burst_ids),
        processingLevel="CSLC-STATIC",
    )

    logger.info(f"Static layer files to download: {len(results)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results.download(path=temp_dir, processes=n_workers)

    # Get static layer paths
    list_static_files = [
        Path(f'{temp_dir}/{results[ii].properties["fileName"]}')
        for ii in range(len(results))
    ]

    # Stitch static layers: generate los_east.tif, los_north.tif, layover_shadow.tif
    logger.info(f"Stitching burst static layers to {output_dir}")
    _ = stitch_geometry_layers(list_static_files, output_dir=output_dir)

    # Generate los_up.tif
    los_east, los_east_atr = open_image(output_dir / "los_east.tif")
    los_north, _ = open_image(output_dir / "los_north.tif")

    # Ensure los_east and los_north are on the same grid as DISP
    geo_info = get_geospatial_info(disp_path)
    atr = {"rows": geo_info.rows, "cols": geo_info.cols}

    for txt, data in zip(["los_east.tif", "los_north.tif"], [los_east, los_north]):
        logger.info(f"Reprojecting {(output_dir / txt).name} to DISP grid")
        reproject_raster(
            output_name=output_dir / txt,
            src_data=data,
            atr=atr,
            src_transform=los_east_atr["gt"],
            dst_transform=geo_info.transform,
            src_crs=los_east_atr["crs"],
            dst_crs=geo_info.crs,
            dtype="float32",
            resampling_mode=Resampling.bilinear,
        )

    logger.info(f"Writing los_up {output_dir}")
    # Read los_east and los_north
    los_east, los_east_atr = open_image(output_dir / "los_east.tif")
    los_north, _ = open_image(output_dir / "los_north.tif")

    # Get los_up
    mask = np.ma.masked_equal(los_east, 0).mask
    up = np.sqrt(1 - los_east**2 - los_north**2)
    up = np.ma.masked_array(up, mask=mask).filled(0)

    # Write los_up to file
    write_geotiff(
        output_dir / "los_up.tif",
        up,
        los_east_atr["bounds"],
        epsg=los_east_atr["crs"].to_epsg(),
    )

    # Delete temporary directory
    shutil.rmtree(temp_dir)

    return {
        "los_east": output_dir / "los_east.tif",
        "los_north": output_dir / "los_north.tif",
        "los_up": output_dir / "los_up.tif",
        "layover_shadow": output_dir / "layover_shadow.tif",
    }


def los_unit2inc_azimuth(
    los_east: Union[str, Path],
    los_north: Union[str, Path],
    output_dir: Union[str, Path],
) -> dict:
    """Convert line-of-sight (LOS) unit vectors to incidence and azimuth angles.

    This function processes east and north LOS unit vector components to derive
    the incidence and azimuth angles and saves the output to the specified directory.

    Parameters
    ----------
    los_east : str | Path
        Path to the file containing the east component of the LOS unit vector.
    los_north : str | Path
        Path to the file containing the north component of the LOS unit vector.
    output_dir : str | Path
        Directory where the output incidence and azimuth angle files will be saved.

    Returns
    -------
    dict
        Dictionary containing metadata or paths of the generated incidence and
        azimuth angle files.

    """
    # Read los_east and los_north
    try:
        los_ew, los_east_atr = open_image(los_east)
    except FileNotFoundError:
        raise ValueError(f"File {los_east} not found")

    try:
        los_ns, _ = open_image(los_north)
    except FileNotFoundError:
        raise ValueError(f"File {los_north} not found")

    # Create output directory if does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get los_up
    up = np.sqrt(1 - los_ew**2 - los_ns**2)
    # Get azimuth and incidence angles
    az_angle = -1 * np.rad2deg(np.arctan2(los_ew, los_ns)) % 360
    incidence_angle = np.rad2deg(np.arccos(up))

    # Write azimuth and incidence angles to file
    logger.info(f"Writing azimuth and incidence angles to {output_dir}")
    write_geotiff(
        output_dir / "azimuth.tif",
        az_angle,
        los_east_atr["bounds"],
        epsg=los_east_atr["crs"].to_epsg(),
    )
    write_geotiff(
        output_dir / "incidence.tif",
        incidence_angle,
        los_east_atr["bounds"],
        epsg=los_east_atr["crs"].to_epsg(),
    )
    return {
        "inc_angle": output_dir / "incidence.tif",
        "az_angle": output_dir / "azimuth.tif",
    }


def download_dem(disp_path: Union[str, Path], output_dir: Union[str, Path]) -> Path:
    """Download a Digital Elevation Model (DEM) for the given DISP dataset.

    Retrieves the DEM corresponding to the provided DISP dataset and saves it
    in the specified output directory.

    Parameters
    ----------
    disp_path : str | Path
        Path to the DISP dataset used to determine the required DEM coverage.
    output_dir : str | Path
        Directory where the downloaded DEM file will be saved.

    Returns
    -------
    Path
        Path to the downloaded DEM file.

    """
    # Create output directory if does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get geospatial info
    geo_info = get_geospatial_info(disp_path)

    # Init projection converter
    transformer = Transformer.from_crs(
        f"EPSG:{geo_info.crs.to_epsg()}", "EPSG:4326", always_xy=True
    )

    # Get bounds in geographic coordinates
    snwe = np.zeros((4))
    left, bottom, right, top = geo_info.bounds
    snwe[2], snwe[1] = transformer.transform(left - 1e3, top + 1e3)
    snwe[3], snwe[0] = transformer.transform(right + 1e3, bottom - 1e3)
    bounds = [snwe[2], snwe[0], snwe[3], snwe[1]]

    # Download dem
    logger.info(f"Downloading DEM for bounds: {np.squeeze(bounds)}")
    atr = {"rows": geo_info.rows, "cols": geo_info.cols}
    dem_data, dem_attr = dem_stitcher.stitch_dem(
        bounds,
        dem_name="glo_30",
        dst_ellipsoidal_height=True,
        dst_area_or_point="Point",
    )

    logger.info(f"Saving DEM: {output_dir / 'dem.tif'}")
    reproject_raster(
        output_name=output_dir / "dem.tif",
        src_data=dem_data,
        atr=atr,
        src_transform=dem_attr["transform"],
        dst_transform=geo_info.transform,
        src_crs=dem_attr["crs"],
        dst_crs=geo_info.crs,
        dtype="float32",
        resampling_mode=Resampling.bilinear,
    )
    return output_dir / "dem.tif"


def download_water_mask(
    disp_path: Union[str, Path], output_dir: Union[str, Path]
) -> Path:
    """Download a water mask for the given DISP dataset.

    Retrieves a water mask corresponding to the provided DISP dataset and saves it
    in the specified output directory.

    Parameters
    ----------
    disp_path : str | Path
        Path to the DISP dataset used to determine the required water mask coverage.
    output_dir : str | Path
        Directory where the downloaded water mask file will be saved.

    Returns
    -------
    Path
        Path to the downloaded water mask file.

    """
    # Create output directory if does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get geospatial info
    geo_info = get_geospatial_info(disp_path)

    # Init projection converter
    transformer = Transformer.from_crs(
        f"EPSG:{geo_info.crs.to_epsg()}", "EPSG:4326", always_xy=True
    )

    # Get bounds in geographic coordinates
    snwe = np.zeros((4))
    left, bottom, right, top = geo_info.bounds
    snwe[2], snwe[1] = transformer.transform(left - 1e3, top + 1e3)
    snwe[3], snwe[0] = transformer.transform(right + 1e3, bottom - 1e3)
    bounds = [snwe[2], snwe[0], snwe[3], snwe[1]]

    # Download water mask
    logger.info(f"Downloading water mask for bounds: {np.squeeze(bounds)}")
    atr = {"rows": geo_info.rows, "cols": geo_info.cols}
    mask_data, mask_attr = tile_mate.get_raster_from_tiles(
        bounds, tile_shortname="esa_world_cover_2021"
    )

    # Make byte mask
    mask_data[mask_data == 80] = 0
    mask_data[mask_data != 0] = 1
    mask_data = mask_data.astype("byte")

    logger.info(f"Saving Water Mask: {output_dir / 'water_mask.tif'}")
    reproject_raster(
        output_name=output_dir / "water_mask.tif",
        src_data=mask_data,
        atr=atr,
        src_transform=mask_attr["transform"],
        dst_transform=geo_info.transform,
        src_crs=mask_attr["crs"],
        dst_crs=geo_info.crs,
        dtype="uint8",
        resampling_mode=Resampling.nearest,
    )
    return output_dir / "water_mask.tif"
