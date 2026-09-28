from pathlib import Path
import numpy as np
import xarray as xr
import rioxarray
from scipy.ndimage import convolve
from scipy.signal import find_peaks, detrend
from scipy.interpolate import interp1d
from typing import Optional, Dict, Any


script_dir = Path(__file__).resolve().parent
data_dir = script_dir / ".." / "data" 
data_topo = script_dir / ".." / "data" / "topography"
data_validation = script_dir / ".." / "data" / "validation_points"
mask_dir = data_dir / "masks"
out_dir = script_dir / ".." / "data" / "output"
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


def detrend_1d_brutal(arr1d):
    """
    Detrend a 1D timeserie with linear adjustement,
    by equaling the first and last values (cyclic continuity),
    while keeping the same average.
    """
    # Si tout est NaN, retourne la série telle quelle
    if np.all(np.isnan(arr1d)):
        return arr1d

    # Interpoler les NaN pour pouvoir appliquer detrend
    arr = arr1d.copy()
    nans = np.isnan(arr)
    if np.any(nans):
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])

    # Calculer la moyenne originale
    vel_mean = np.nanmean(arr)

    # Calculer la tendance linéaire entre la première et la dernière valeur
    x = np.arange(len(arr))
    first_val = arr[0]
    last_val = arr[-1]
    trend = np.linspace(first_val, last_val, len(arr))

    # Detrender la série en soustrayant la tendance et réaujster la moyenne
    vel_detrended_cyclic = arr - trend + vel_mean

    return vel_detrended_cyclic


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


def fft_filter(x, sample_step_days=5, cutoff_days=210):
    n = x.shape[0]
    freqs = np.fft.fftfreq(n, d=sample_step_days)
    x_fft = np.fft.fft(x, axis=0)
    x_fft[np.abs(freqs) > 1 / cutoff_days] = 0
    return np.real(np.fft.ifft(x_fft, axis=0))


def create_avg_year_smooth(vel_ts, doys):
    vel_avg_year = []

    for d in np.arange(1, 366):
        delta = np.abs(doys - d)
        delta = np.minimum(delta, 365 - delta)  # cyclic distance

        mask = delta <= 2 # mask 1d +-2 jours
        v = vel_ts.isel(mid_date=mask).mean(dim='mid_date')

        vel_avg_year.append(v.values)

    vel_avg_year = np.stack(vel_avg_year)   # (365, ny, nx)

    window = 5  # smooth over +-2 day
    vel_avg_year_smooth = np.empty_like(vel_avg_year)

    for iy in range(vel_avg_year.shape[1]):
        for ix in range(vel_avg_year.shape[2]):
            vel = vel_avg_year[:, iy, ix]
            # cyclic extension to avoid the bounds
            v_ext = np.concatenate([vel[-(window//2):], vel, vel[:window//2]])
            v_smooth = np.convolve(v_ext, np.ones(window)/window, mode='valid')
            vel_avg_year_smooth[:, iy, ix] = v_smooth

    return vel_avg_year_smooth



### ----- Reprojecting dem, slope, masks and timeseries -----


def load_and_align_raster(path: Path, ref_grid: xr.DataArray) -> xr.DataArray:
    """Charge et aligne un raster GeoTIFF sur une grille de référence."""
    da = rioxarray.open_rasterio(path).squeeze()
    return da.rio.reproject_match(ref_grid)



def compute_slope(elevation, dx, dy):
    """Calcule la pente à partir d'un raster d'élévation."""
    dy_elev, dx_elev = np.gradient(elevation)
    # Diviser par dx et dy pour obtenir des gradients en unités d'élévation par unité de distance
    dy_elev /= dy
    dx_elev /= dx
    print(f"Gradient dx: Min: {dx_elev.min()}, Max: {dx_elev.max()}")
    print(f"Gradient dy: Min: {dy_elev.min()}, Max: {dy_elev.max()}")
    slope_rad = np.arctan(np.sqrt(dx_elev**2 + dy_elev**2))
    return np.degrees(slope_rad)




### ----- Plotting tools -----

def moving_average(y, x, step, window):
    centers = np.arange(np.nanmin(x), np.nanmax(x)+step, step)
    result = np.full_like(centers, np.nan, dtype=float)
    for i, c in enumerate(centers):
        mask = (x >= c - (window/2)) & (x <= c + (window/2))
        if np.any(mask):
            result[i] = np.nanmean(y[mask])
    return centers, result


def moving_median(y, x, step, window):
    centers = np.arange(np.nanmin(x), np.nanmax(x)+step, step)
    result = np.full_like(centers, np.nan, dtype=float)
    for i, c in enumerate(centers):
        mask = (x >= c - (window/2)) & (x <= c + (window/2))
        if np.any(mask):
            result[i] = np.nanmedian(y[mask])
    return centers, result



### ----- Friction law analysis -----

def power_law(u_bed, As, m=3):
    tau_b = (u_bed/As)**(1/m)
    return tau_b


def cavitation_law(u_bed, CN, q, As, m=3): # no rate weakening
    alpha = ((q-1)**(q-1))/(q**q)
    chi = u_bed /(As*(CN)**m)
    tau_b = (CN)*(chi/(1+alpha*chi**q))**(1/m)
    
    if q != 1:  # Avoid division by zero error
        try:
            # Find u_bed_max so that tau_b is maximal and define an asymptote for this value
            u_bed_max = (As * CN**m) * (1 / (alpha * (q-1)))**(1/q)
            tau_b = np.where(u_bed > u_bed_max, CN, tau_b)
        except ZeroDivisionError:
            pass

    return tau_b