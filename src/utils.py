import numpy as np
import pandas as pd
import geoutils as gu
from scipy.ndimage import convolve
from scipy.signal import find_peaks, detrend
from scipy.interpolate import interp1d
from pathlib import Path
import rasterio
import xarray as xr

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


def peak_and_index(arr_1d):
    """Returns [max_val, max_idx, min_val, min_idx] for a 1D signal."""    
    out = np.full(4, np.nan, dtype=np.float64)  # initialise a table of size 4
    if np.all(np.isnan(arr_1d)):
        return out

    peaks, _ = find_peaks(arr_1d)
    troughs, _ = find_peaks(-arr_1d)

    if len(peaks) > 0:
        peak_vals = arr_1d[peaks]
        max_peak_idx = peaks[np.nanargmax(peak_vals)]
        out[0] = arr_1d[max_peak_idx]
        out[1] = max_peak_idx

    if len(troughs) > 0:
        trough_vals = arr_1d[troughs]
        min_trough_idx = troughs[np.nanargmin(trough_vals)]
        out[2] = arr_1d[min_trough_idx]
        out[3] = min_trough_idx

    return out


def save_geotiff(da, path, epsg=None):
    """Saves a DataArray into GeoTIFF."""
    if epsg is not None:
        da = da.rio.write_crs(f"EPSG:{epsg}")

    # Sorts Y to ensure it matches conventionnal raster order
    if "y" in da.coords and da.y[0] < da.y[-1]:
        da = da.sortby("y", ascending=False)

    da.rio.to_raster(path)

    

def idx_to_doy(idx_da, doy_array):
    idx_vals = idx_da.values.astype(float)
    valid = (
        ~np.isnan(idx_vals) &
        (idx_vals >= 0) &
        (idx_vals < len(doy_array))
    )
    out = np.full_like(idx_vals, np.nan, dtype=float)
    out[valid] = doy_array[idx_vals[valid].astype(int)]
    return xr.DataArray(
        out,
        coords=idx_da.coords,
        dims=idx_da.dims,
        name=(idx_da.name or "idx") + "_doy"
    )


def read_tif(path, band=1):
    """Returns a numpy array (with NaN where NoData) from a GeoTIFF."""
    with rasterio.open(path) as src:
        arr = src.read(band).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    return arr



def tif_to_xrda(path, band=1, coords=None, dims=("y","x")):
    """Returns a xarray.DataArray (mask if NoData) from a tif."""
    with rasterio.open(path) as src:
        arr = src.read(band).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    return xr.DataArray(arr, coords=coords, dims=dims)



def build_melt_cycle(dem_da, unique_cycles, dates, temps, cycles):
    """
    Estimate daily melt rate (m w.e.) from SAFRAN temperature at 2400 m 
    using lapse rate correction and degree-day factor.
    temp_2400: xarray DataArray of daily temperatures at 2400 m
    z: xarray DataArray of elevations
    """
    ncycle = unique_cycles.size
    ny, nx = dem_da.shape
    sums = {c: np.zeros((ny, nx), dtype=np.float64) for c in unique_cycles}
    counts = {c: np.zeros((ny, nx), dtype=np.int32) for c in unique_cycles}

    # boucle sur dates (mémoire O(ny*nx))
    for date, temp2400, cyc in zip(dates, temps, cycles):
        # calcule T_local = temp2400 - lapse*(z-2400)
        # dem_da.data is 2D numpy (si xarray, .values)
        T_local = temp2400 - 0.0065 * (dem_da.values - 2400.0)

        # calcule melt (m w.e. per day), 0 si <0
        melt_2d = np.where(T_local > 0, 0.009 * T_local, 0.0)

        # appliquer mask (évite stocker valeurs hors zone)
        valid = np.isfinite(melt_2d)

        # agregation
        sums[cyc][valid] += melt_2d[valid]
        counts[cyc][valid] += 1

    # construire le résultat: moyenne par cycle (y,x)
    melt_cycle_mean = np.full((ncycle, ny, nx), np.nan, dtype=np.float32)
    cycle_list = sorted(unique_cycles)
    for i,c in enumerate(cycle_list):
        m = sums[c]
        cnt = counts[c]
        ok = cnt > 0
        melt_cycle_mean[i, ok] = (m[ok] / cnt[ok]).astype(np.float32)

    return melt_cycle_mean, cycle_list



def moving_average(y, x, step, window):
    centers = np.arange(np.nanmin(x), np.nanmax(x)+step, step)
    result = np.full_like(centers, np.nan, dtype=float)
    for i, c in enumerate(centers):
        mask = (x >= c - (window/2)) & (x <= c + (window/2))
        if np.any(mask):
            result[i] = np.nanmean(y[mask])
    return centers, result


from rasterio.warp import reproject, Resampling

def load_and_interp_geotiff_to_grid(tif_path, ds, target_epsg="32632", resampling="bilinear"):
    """
    Charge le DEM, le reprojette en EPSG target, et l'interpole sur la grille du Dataset TICOI.
    """

    with rasterio.open(tif_path) as src:
        dem = src.read(1)  # matrice
        src_transform = src.transform
        src_crs = src.crs

    x = ds.x.values
    y = ds.y.values[::-1] # y décroissant

    from rasterio.transform import from_origin
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dst_transform = from_origin(x.min(), y.max(), dx, -dy)
    dst_crs = f"EPSG:{target_epsg}"

    # grid size
    dst_height = len(y)
    dst_width = len(x)

    dem_on_grid = np.zeros((dst_height, dst_width), dtype=np.float32)

    # reprojette et interpole
    reproject(source=dem, destination=dem_on_grid, src_transform=src_transform, src_crs=src_crs, dst_transform=dst_transform, dst_crs=dst_crs, resampling=getattr(Resampling, resampling))

    return dem_on_grid[::-1, :]


def compute_slope(dem, dx, dy):
    """
    Pente en radians.
    """
    dzdx = np.gradient(dem, axis=1) / dx
    dzdy = -np.gradient(dem, axis=0) / dy

    slope_rad = np.sqrt(dzdx**2 + dzdy**2)
    slope_deg = np.degrees(slope_rad)
    return slope_deg



import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

def load_geotiff_as_da(tif_path, ds_ref, target_epsg=None, resampling="bilinear", name="dem"):
    """
    Charge un GeoTIFF et le reprojette/interpole sur la grille d'un xarray.Dataset de référence.
    Retourne un DataArray avec coords et CRS alignés.
    
    Parameters
    ----------
    tif_path : str or Path
        Chemin vers le GeoTIFF à charger.
    ds_ref : xarray.Dataset or DataArray
        Dataset de référence pour la grille.
    target_epsg : str or int, optional
        EPSG cible. Si None, prend le CRS de ds_ref.
    resampling : str, default "bilinear"
        Méthode de resampling pour reproject.
    name : str
        Nom du DataArray retourné.
    """
    # Grille de référence
    x = ds_ref.x.values
    y = ds_ref.y.values[::-1]  # y décroissant
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dst_transform = from_origin(x.min(), y.max(), dx, -dy)
    dst_crs = f"EPSG:{target_epsg}" if target_epsg else ds_ref.rio.crs
    
    dst_shape = (len(y), len(x))
    dst_array = np.full(dst_shape, np.nan, dtype=np.float32)
    
    # Charger GeoTIFF source
    with rasterio.open(tif_path) as src:
        src_array = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
    
    # Reprojection / interpolation
    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling[resampling],
        dst_nodata=np.nan
    )
    
    # Retourner en DataArray xarray
    da = xr.DataArray(
        dst_array[::-1, :],  # remettre y dans le bon ordre
        coords={"y": ds_ref.y, "x": ds_ref.x},
        dims=["y", "x"],
        name=name
    )
    
    da.rio.write_crs(dst_crs, inplace=True)
    return da


from scipy.ndimage import uniform_filter
import numpy as np
import xarray as xr

def compute_slope_da(dem_da):
    """
    Calcul de la pente (en degrés) à partir d'un DataArray xarray DEM.
    Assumes une grille régulière (dx, dy constants).
    
    Parameters
    ----------
    dem_da : xarray.DataArray
        DEM déjà aligné sur la grille de référence.

    Returns
    -------
    slope_da : xarray.DataArray
        Pente en degrés, même coords et dims que dem_da.
    """
    # Extraire NumPy array
    dem_arr = dem_da.values.astype(float)
    
    # Résolution spatiale
    dx = np.abs(dem_da.x[1] - dem_da.x[0])
    dy = np.abs(dem_da.y[1] - dem_da.y[0])
    
    # Gradient
    dzdx = np.gradient(dem_arr, axis=1) / dx
    dzdy = -np.gradient(dem_arr, axis=0) / dy  # y décroissant
    
    # Pente en degrés
    slope_deg = np.degrees(np.sqrt(dzdx**2 + dzdy**2))
    
    # Recréer DataArray xarray
    slope_da = xr.DataArray(
        slope_deg,
        coords=dem_da.coords,
        dims=dem_da.dims,
        name="slope"
    )
    
    # Ajouter CRS si présent
    if hasattr(dem_da.rio, "crs") and dem_da.rio.crs is not None:
        slope_da.rio.write_crs(dem_da.rio.crs, inplace=True)
    
    return slope_da
