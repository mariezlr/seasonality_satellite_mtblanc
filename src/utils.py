import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from scipy.signal import find_peaks, detrend
from scipy.interpolate import interp1d
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = script_dir / ".." / "data" 
data_topo = script_dir / ".." / "data" / "topography"
data_validation = script_dir / ".." / "data" / "validation_points"
fig_dir = script_dir / ".." / "figures" 

### ----- Velocity timeseries analysis -----

# Fusionner les datasets en s'assurant que le dernier fichier écrase les anciens en cas de doublons
def merge_datasets(ds_list):
    ds_merged = ds_list[0]
    for ds in ds_list[1:]:
        ds_merged = ds.combine_first(ds_merged)  # Priorité au dernier dataset
    return ds_merged

# Fonction de convolution à appliquer sur chaque tranche (y, x)
def convolve_2d(arr2d, kernel):
    return convolve(arr2d, kernel, mode='nearest')


# Fonction pour détrender une série 1D
def detrend_1d(arr1d):
    # Si tout est NaN, retourne la série telle quelle
    if np.all(np.isnan(arr1d)):
        return arr1d
    
    # Interpoler les NaN pour pouvoir appliquer detrend
    arr = arr1d.copy()
    nans = np.isnan(arr)
    if np.any(nans):
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    
    vel_detrended = detrend(arr, type="linear")
    vel_mean = np.nanmean(arr)
    
    # Appliquer detrend
    return vel_detrended + vel_mean

## Interpoler les Nan pour que le filtrage fonctionne
def interp_1d_fill(values, times):
    if np.all(np.isnan(values)):
        return values

    # Convertir les dates en float (ex: jours depuis origine)
    origin = np.datetime64("1970-01-01")
    times_float = (times - origin) / np.timedelta64(1, "D")

    valid = ~np.isnan(values)
    interp = interp1d(
        times_float[valid], values[valid],
        bounds_error=False,
        fill_value="extrapolate"
    )
    return interp(times_float)


def fft_filter(x, sample_step_days=210, cutoff_days=5):
    n = x.shape[0]
    freqs = np.fft.fftfreq(n, d=sample_step_days)
    x_fft = np.fft.fft(x, axis=0)
    x_fft[np.abs(freqs) > 1 / cutoff_days] = 0
    return np.real(np.fft.ifft(x_fft, axis=0))




