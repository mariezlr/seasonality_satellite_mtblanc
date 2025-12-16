from utils import *
from data_exploration import *
import numpy as np
import xarray as xr
import rioxarray
from scipy.signal import welch
import matplotlib.pyplot as plt


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


def compute_total_mask_and_export(output_file_path, xcount, vel_result, shadow_raster_path, min_valid_obs=80, threshold_xcount=100, threshold_shadow=50, snr_threshold=15):
    # Ensure velocity result has CRS and x/y coordinates for reprojection
    if not hasattr(vel_result, "rio"):
        vel_result = vel_result.rio.write_crs("EPSG:32632", inplace=False)
    elif vel_result.rio.crs is None:
        vel_result.rio.write_crs("EPSG:32632", inplace=True)

    base_mask = vel_result.notnull().any(dim='mid_date')
    base_mask = base_mask.astype('uint8')
    base_mask = base_mask.rio.write_crs("EPSG:32632", inplace=True)
    base_mask = base_mask.rio.reproject_match(vel_result)
    base_mask.rio.to_raster(output_file_path / "base_mask.tif")


    mask_xcount = create_mask_xcount(xcount, min_valid_obs, threshold_xcount)
    mask_xcount = mask_xcount.astype('uint8')
    mask_xcount = mask_xcount.rio.write_crs("EPSG:32632", inplace=True)
    mask_xcount = mask_xcount.rio.reproject_match(vel_result)
    mask_xcount.rio.to_raster(output_file_path / "mask_xcount.tif")


    mask_shadow = create_mask_shadow(vel_result, shadow_raster_path, threshold_shadow)
    mask_shadow = mask_shadow.astype('uint8')
    mask_shadow = mask_shadow.rio.write_crs("EPSG:32632", inplace=True)
    mask_shadow = mask_shadow.rio.reproject_match(vel_result)
    mask_shadow.rio.to_raster(output_file_path / "mask_shadow.tif")


    mask_snr, snr_map = create_mask_snr(vel_result, snr_threshold)
    mask_snr = mask_snr.astype('uint8')
    mask_snr = mask_snr.rio.write_crs("EPSG:32632", inplace=True)
    mask_snr = mask_snr.rio.reproject_match(vel_result)
    mask_snr.rio.to_raster(output_file_path / "mask_snr.tif")


    mask_total = base_mask & mask_xcount & mask_shadow & mask_snr
    mask_total.rio.to_raster(output_file_path / "mask_total.tif")

    return mask_total, base_mask, mask_xcount, mask_shadow, mask_snr



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





