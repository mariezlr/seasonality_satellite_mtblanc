from utils import *
from data_exploration import *
import numpy as np
import xarray as xr
import rioxarray
from pathlib import Path

def compute_peaks_and_export(velocity_cycle_mean, out_dir, epsg=32632):
    """
    Calculates peaks (max/min values + indices) and exports GeoTIFFs.
    Also computes DOY rasters and exports them.
    velocity_cycle_mean must have dim 'cycle'.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = xr.apply_ufunc(
        peak_and_index,
        velocity_cycle_mean,
        input_core_dims=[['cycle']],
        output_core_dims=[['stat']],
        vectorize=True,
        output_dtypes=[float],
        output_sizes={'stat': 4}
    )

    datasets = {
        "max_peak_vals": result.sel(stat=0).rename("max_peak_vals"),
        "max_peak_idx": result.sel(stat=1).astype(int).rename("max_peak_idx"),
        "min_peak_vals": result.sel(stat=2).rename("min_peak_vals"),
        "min_peak_idx": result.sel(stat=3).astype(int).rename("min_peak_idx"),
    }

    # Save GeoTIFFs (peaks)
    for name, da in datasets.items():
        save_geotiff(da, f"{out_dir}/{name}.tif", epsg=epsg)


    # Compute doy from cycle indices
    doy_array = velocity_cycle_mean['doy_approx'].values

    max_peak_doy = idx_to_doy(datasets["max_peak_idx"], doy_array)
    min_peak_doy = idx_to_doy(datasets["min_peak_idx"], doy_array)

    # Save doy rasters
    save_geotiff(max_peak_doy, f"{out_dir}/max_peak_doy.tif", epsg=epsg)
    save_geotiff(min_peak_doy, f"{out_dir}/min_peak_doy.tif", epsg=epsg)

    # Add to returned dict
    datasets["max_peak_doy"] = max_peak_doy
    datasets["min_peak_doy"] = min_peak_doy

    return datasets


count_short_before_max = 0
count_no_local_max = 0

def index_of_pre_peak_derivative(arr_1d):
    global count_short_before_max, count_no_local_max

    out = np.full(2, np.nan, dtype=float)

    if np.all(np.isnan(arr_1d)) or len(arr_1d) < 3:
        return out    
   
    #d2 = arr_1d[2:] - 2*arr_1d[1:-1] + arr_1d[:-2]  # Dérivée seconde discrète
    #sign_change = np.where(np.diff(np.sign(d2)) != 0)[0] # Indices où la dérivée seconde change de signe
    #selected_idx = sign_change[0] + 1   # premier changement de concavité

    # Trouver le minimum global de la vitesse
    idx_max = np.nanargmax(arr_1d)

    if idx_max <= 2:  # pas assez de points avant le max
        count_short_before_max += 1
        return out

    # Calculer la dérivée première discrète sur la portion avant le pic
    deriv = np.diff(arr_1d[:idx_max])

    # Trouver tous les maxima locaux de la dérivée
    local_max = np.where((deriv[1:-1] > deriv[:-2]) & (deriv[1:-1] > deriv[2:]))[0] + 1

    if len(local_max) == 0:
        count_no_local_max += 1
        return out

    # On prend le dernier maximum local avant le pic
    selected_idx = local_max[-1]

    out[0] = arr_1d[selected_idx]
    out[1] = selected_idx

    return out


def compute_inflex_and_export(velocity_cycle_mean, out_dir, epsg=32632):
    """
    Calculates peaks (max/min values + indices) and exports GeoTIFFs.
    Also computes DOY rasters and exports them.
    velocity_cycle_mean must have dim 'cycle'.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inflex_points = xr.apply_ufunc(
        index_of_pre_peak_derivative,
        velocity_cycle_mean,
        input_core_dims=[['cycle']],
        output_core_dims=[['stat']],
        vectorize=True,
        output_dtypes=[float],
        output_sizes={'stat': 2}
    )

    datasets = {
        "inflex_vals": inflex_points.sel(stat=0).rename("inflex_vals"),
        "inflex_idx": inflex_points.sel(stat=1).astype(int).rename("inflex_idx"),
    }

    # Save GeoTIFFs (peaks)
    for name, da in datasets.items():
        save_geotiff(da, f"{out_dir}/{name}.tif", epsg=epsg)


    # Compute doy from cycle indices
    doy_array = velocity_cycle_mean['doy_approx'].values

    inflex_doy = idx_to_doy(datasets["inflex_idx"], doy_array)

    # Save doy rasters
    save_geotiff(inflex_doy, f"{out_dir}/inflex_doy.tif", epsg=epsg)

    # Add to returned dict
    datasets["inflex_doy"] = inflex_doy

    return datasets




