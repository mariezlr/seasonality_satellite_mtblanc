from utils import *
from data_exploration import *
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
from scipy.signal import welch
from pathlib import Path

# base_mask = vel_result.notnull().any(dim='mid_date')


def create_mask_xcount(xcount, min_valid_obs=80, threshold_xcount=100):
    """
    Create a mask based on number of valid xcount observations.

    xcount : xarray.DataArray. Observed xcount per date.
    min_valid_obs : minimal number of valid dates
    threshold_xcount : threshold to consider that an obs is "valid"
    """
    valid_obs = xcount >= threshold_xcount    # bool for each date   
    valid_obs_count = valid_obs.sum(dim='mid_date')    # number of valid dates for each pixel   
    mask = valid_obs_count >= min_valid_obs    
    return mask



def create_mask_shadow(vel_result, shadow_raster_path, threshold_shadow=50):
    """
    Mask pixels based on a shadow raster. Pixels below threshold are valid.

    vel_result : Reference velocity raster for reprojection. (xarray.DataArray)
    shadow_raster_path : Path to the shadow map raster.
    threshold_shadow : Maximum allowed shadow value for valid pixels.
    """
    raster = rioxarray.open_rasterio(shadow_raster_path).squeeze()
    raster_utm = raster.rio.reproject_match(vel_result)
    
    mask = raster_utm < threshold_shadow
    return mask


def create_mask_velavg(vel_result, threshold_velavg=30):
    """
    Mask pixels based on their mean velocity. Pixels above 30m/yr are valid. Pixels below are not exploitable for seasonality analysis.

    vel_result : Reference velocity raster for reprojection. (xarray.DataArray)
    threshold_velavg: Threshold for mean velocity (default: 30 m/yr)
    """
    
    mask = vel_result.mean(dim="mid_date") > threshold_velavg
    return mask


def seasonal_snr_welch(ts_1d, time_step_days=6):
    """
    Compute the seasonal signal-to-noise ratio using Welch's method.
    Returns NaN for invalid time series.
    """
    ts_1d = np.asarray(ts_1d)
    
    if np.all(np.isnan(ts_1d)) or len(ts_1d) < 10:
        return np.nan
    
    ts_1d = np.nan_to_num(ts_1d - np.nanmean(ts_1d))
    
    fs = 1 / time_step_days
    f_annual = 1 / 365
    band = 0.1 * f_annual
    
    f, Pxx = welch(ts_1d, fs=fs, nperseg=len(ts_1d), scaling='spectrum')
    A = np.sqrt(2 * Pxx)
    
    idx_annual = np.argmin(np.abs(f - f_annual))
    amp_annual = A[idx_annual]
    
    noise_mask = np.abs(f - f_annual) > band
    noise = np.nanmean(A[noise_mask])
    
    if noise <= 0:
        return np.nan
    
    return amp_annual / noise


def create_mask_snr(vel_result, snr_threshold=15):
    """
    Compute SNR map and return a mask where SNR exceeds threshold.
    """
    snr_map = xr.apply_ufunc(
        seasonal_snr_welch,
        vel_result,
        input_core_dims=[['mid_date']],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    mask = snr_map > snr_threshold
    return mask, snr_map



def compute_total_mask_and_export(
    output_dir: Path,
    xcount: xr.DataArray,
    vel_result: xr.DataArray,
    shadow_raster_path: Path,
    min_valid_obs: int = 80,
    threshold_xcount: int = 100,
    threshold_shadow: int = 50,
    threshold_velavg: int = 20,
    snr_threshold: int = 15,
) -> tuple[xr.DataArray, ...]:
    """
    Compute and export a set of masks (base, xcount, shadow, SNR, and total) as GeoTIFFs.

    Args:
        output_dir: Directory where masks will be saved.
        xcount: DataArray containing the xcount data.
        vel_result: DataArray containing velocity results.
        shadow_raster_path: Path to the shadow raster file.
        min_valid_obs: Minimum number of valid observations for xcount mask.
        threshold_xcount: Threshold for xcount mask.
        threshold_shadow: Threshold for shadow mask.
        snr_threshold: Threshold for SNR mask.

    Returns:
        Tuple of (mask_total, base_mask, mask_xcount, mask_shadow, mask_snr) as xarray.DataArray.
    """
    # Ensure output_dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure vel_result has CRS and x/y coordinates for reprojection
    if not hasattr(vel_result, "rio"):
        raise ValueError("vel_result must be a rio-accessible xarray.DataArray.")
    if vel_result.rio.crs is None:
        vel_result.rio.write_crs("EPSG:32632", inplace=True)

    # --- Base mask ---
    base_mask = vel_result.notnull().any(dim="mid_date").astype("uint8")
    base_mask = _prepare_and_export_mask(
        base_mask,
        vel_result,
        output_dir / "base_mask.tif",
        title="Base Mask",
        description="Mask of pixels with at least one valid observation.",
    )

    # --- Xcount mask ---
    mask_xcount = create_mask_xcount(xcount, min_valid_obs, threshold_xcount)
    mask_xcount = _prepare_and_export_mask(
        mask_xcount,
        vel_result,
        output_dir / "mask_xcount.tif",
        title="Xcount Mask",
        description=f"Mask of pixels with xcount >= {threshold_xcount} and at least {min_valid_obs} valid observations.",
    )

    # --- Shadow mask ---
    mask_shadow = create_mask_shadow(vel_result, shadow_raster_path, threshold_shadow)
    mask_shadow = _prepare_and_export_mask(
        mask_shadow,
        vel_result,
        output_dir / "mask_shadow.tif",
        title="Shadow Mask",
        description=f"Mask of pixels with shadow value <= {threshold_shadow}.",
    )


    # --- Velavg mask ---
    mask_velavg = create_mask_velavg(vel_result, threshold_velavg)
    mask_velavg = _prepare_and_export_mask(
        mask_velavg,
        vel_result,
        output_dir / "mask_velavg.tif",
        title="Velavg Mask",
        description=f"Mask of pixels with average velocity <= {threshold_velavg}.",
    )


    # --- SNR mask ---
    mask_snr, _ = create_mask_snr(vel_result, snr_threshold)
    mask_snr = _prepare_and_export_mask(
        mask_snr,
        vel_result,
        output_dir / "mask_snr.tif",
        title="SNR Mask",
        description=f"Mask of pixels with SNR >= {snr_threshold}.",
    )

    # --- Total mask ---
    mask_total = base_mask & mask_xcount & mask_shadow & mask_velavg & mask_snr
    mask_total.attrs = {
        "title": "Total Mask",
        "description": "Combined mask from base, xcount, shadow, velavg, and SNR masks.",
    }
    mask_total.rio.write_crs(vel_result.rio.crs, inplace=True)
    mask_total.rio.to_raster(
        output_dir / "mask_total.tif",
        dtype="uint8",
        nodata=0,
        compress="LZW",
    )

    return mask_total, base_mask, mask_xcount, mask_shadow, mask_velavg, mask_snr

def _prepare_and_export_mask(
    mask: xr.DataArray,
    reference: xr.DataArray,
    output_path: Path,
    title: str,
    description: str,
) -> xr.DataArray:
    """
    Prepare a mask for export: set CRS, reproject, add metadata, and export as GeoTIFF.

    Args:
        mask: Input mask as xarray.DataArray.
        reference: Reference DataArray for CRS and reprojection.
        output_path: Path to save the mask.
        title: Title for the mask metadata.
        description: Description for the mask metadata.

    Returns:
        xarray.DataArray: The prepared mask.
    """
    mask = mask.astype("uint8")
    mask.attrs = {
        "title": title,
        "description": description,
    }
    mask.rio.write_crs(reference.rio.crs, inplace=True)
    mask = mask.rio.reproject_match(reference)
    mask.rio.to_raster(
        output_path,
        dtype="uint8",
        nodata=0,
        compress="LZW",
    )
    return mask




# Quick visualization
def show_mask(mask, title="Mask"):
    """Quick plot of a mask using matplotlib."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.colorbar()
    plt.show()


def mask_stats(mask, velocities):
    """
    Print basic statistics of the mask.
    
    Returns Total number of pixels in the velocity grid and Number of pixels retained by the mask.
    """
    n_total_pix = velocities.isel(mid_date=0).size
    n_valid_pix = mask.sum().item()
    return n_total_pix, n_valid_pix



# Masque zones statiques
from rasterio.features import rasterize

def create_mask_stable_areas(vel_result, stable_areas_geospatial_path):
    """
    Compute a boolean mask for stable areas from a geospatial file.
    The mask is reprojected to match the reference raster (vel_result).

    vel_result : Reference velocity raster (xarray.DataArray)
    stable_areas_geospatial_path : Path to the geospatial file (.gpkg)
    """
    gdf = gpd.read_file(stable_areas_geospatial_path)

    # Delete nul, empty and invalid geometries
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    gdf["geometry"] = gdf.buffer(0)
    gdf = gdf[gdf.is_valid]  

    # Reprojection + alignment
    gdf = gdf.to_crs(vel_result.rio.crs)

    transform = vel_result.rio.transform()
    out_shape = (vel_result.sizes["y"], vel_result.sizes["x"])

    # Rasterize
    mask_array = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    # Convert to boolean DataArray
    mask_bool = xr.DataArray(
        mask_array.astype(bool),
        coords={
            "y": vel_result["y"],
            "x": vel_result["x"]},
        dims=("y", "x"),
        name="stable_areas_mask"
    )

    return mask_bool



def compute_mask_stable_areas_and_export(
    output_dir: Path,
    vel_result: xr.DataArray,
    stable_areas_geospatial_path: Path
):
    """
    Compute and export a set of masks (base, xcount, shadow, SNR, and total) as GeoTIFFs.

    Args:
        output_dir: Directory where masks will be saved.
        vel_result: DataArray containing velocity results.

    Returns:
        stable_areas_mask as xarray.DataArray.
    """
    # Ensure output_dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure vel_result has CRS and x/y coordinates for reprojection
    if not hasattr(vel_result, "rio"):
        raise ValueError("vel_result must be a rio-accessible xarray.DataArray.")
    if vel_result.rio.crs is None:
        vel_result.rio.write_crs("EPSG:32632", inplace=True)

    # --- Base mask ---
    mask_stable_areas = create_mask_stable_areas(vel_result, stable_areas_geospatial_path)
    mask_stable_areas = _prepare_and_export_mask(
        mask_stable_areas,
        vel_result,
        output_dir / "mask_stable_areas.tif",
        title="Stable Areas Mask",
        description="Mask of pixels located outside of glaciers",
    )
