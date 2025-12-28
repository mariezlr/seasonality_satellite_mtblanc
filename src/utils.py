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



def export_to_netcdf_with_metadata(
    da: xr.DataArray,
    output_path: Path,
    title: str,
    description: str,
    units: str,
    crs: str = None,
    encoding: dict = None,
    global_attrs: dict = None,
) -> None:
    """
    Exporte un DataArray en NetCDF avec métadonnées, CRS et encodage optimisé.

    Args:
        da: DataArray à exporter.
        output_path: Chemin de sortie pour le fichier NetCDF.
        title: Titre du jeu de données.
        description: Description du contenu.
        units: Unités des données.
        crs: CRS à définir (optionnel, pour les données géospatiales).
        encoding: Dictionnaire d'encodage pour les variables (optionnel).
        global_attrs: Attributs globaux supplémentaires (optionnel).
    """
    # Définir les métadonnées de base
    da.attrs.update({
        "title": title,
        "description": description,
        "units": units,
        "created_by": "Marie ZELLER",
        "institution": "IGE - UGA - CNRS",
        "date_created": str(np.datetime64('now')),
    })

    # Ajouter des attributs globaux supplémentaires si fournis
    if global_attrs:
        da.attrs.update(global_attrs)

    # Définir le CRS si nécessaire (pour les données géospatiales)
    if crs and hasattr(da, "rio"):
        da.rio.write_crs(crs, inplace=True)

    # Encodage par défaut si non fourni
    if encoding is None:
        encoding = {da.name: {"dtype": "float32", "zlib": True, "complevel": 4}}

    # Exporter en NetCDF
    da.to_netcdf(
        output_path,
        engine="netcdf4",
        encoding=encoding,
    )
    print(f"Exported {output_path.name} with metadata and encoding.")


def export_to_geotiff(
    da: xr.DataArray,
    output_path: Path,
    epsg: int = 32632,
    metadata: Optional[Dict[str, Any]] = None,
    nodata: Optional[float] = None,
    dtype: str = "float32",
) -> None:
    """
    Exporte un DataArray en GeoTIFF avec métadonnées et CRS.

    Args:
        da: DataArray à exporter.
        output_path: Chemin de sortie pour le GeoTIFF.
        epsg: Code EPSG du CRS.
        metadata: Dictionnaire de métadonnées supplémentaires.
        nodata: Valeur de "no data".
        dtype: Type de données pour l'export.
    """
    # Définir le CRS si nécessaire
    if not hasattr(da, "rio"):
        raise ValueError("Le DataArray doit être compatible avec rioxarray.")
    if da.rio.crs is None:
        da.rio.write_crs(f"EPSG:{epsg}", inplace=True)

    # Définir les métadonnées par défaut
    default_metadata = {
        "title": da.name.replace("_", " ").title(),
        "description": f"Raster {da.name} computed from analysis.",
        "units": "1" if da.name.endswith(("idx", "doy")) else "m/year",
        "created_by": "Marie ZELLER",
        "institution": "IGE - UGA - CNRS",
        "date_created": str(np.datetime64("now")),
    }

    # Mettre à jour avec les métadonnées fournies
    if metadata:
        default_metadata.update(metadata)

    da.attrs.update(default_metadata)

    # Exporter en GeoTIFF
    da.rio.to_raster(
        output_path,
        dtype=dtype,
        nodata=nodata,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    print(f"Exported {output_path.name} with metadata and CRS EPSG:{epsg}.")



from typing import Optional, Dict, Any
from rasterio.transform import Affine
import tempfile

def export_to_geotiff_with_metadata(
    da: xr.DataArray,
    output_path: Path,
    epsg: int = 32632,
    metadata: Optional[Dict[str, Any]] = None,
    nodata: Optional[float] = None,
    dtype: str = "float32",
) -> None:
    """
    Exporte un DataArray en GeoTIFF avec métadonnées, CRS et options d'export optimisées.

    Args:
        da: DataArray à exporter.
        output_path: Chemin de sortie pour le GeoTIFF.
        epsg: Code EPSG du CRS (par défaut: 32632).
        metadata: Dictionnaire de métadonnées supplémentaires.
        nodata: Valeur de "no data".
        dtype: Type de données pour l'export.
    """
    # Définir le CRS si nécessaire
    if not hasattr(da, "rio"):
        raise ValueError("Le DataArray doit être compatible avec rioxarray.")
    if da.rio.crs is None:
        da.rio.write_crs(f"EPSG:{epsg}", inplace=True)

    # Vérifier et corriger les bornes en Y inversées
    # Sauvegarder temporairement le DataArray pour accéder à ses propriétés de fichier
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)
        da.rio.to_raster(temp_path, dtype=dtype, nodata=nodata)

        with rasterio.open(temp_path) as src:
            transform = src.transform
            bounds = src.bounds

            # Vérifier si les bornes en Y sont inversées (bottom > top)
            if bounds.bottom > bounds.top:
                print("Correction des bornes en Y inversées...")
                # Inverser la transformation en Y
                new_transform = Affine(
                    transform.a,  # Résolution en X
                    transform.b,  # Rotation
                    transform.c,  # Origine en X
                    -transform.e,  # Résolution en Y (inversée)
                    -transform.d,  # Rotation
                    transform.f + (bounds.bottom - bounds.top),  # Origine en Y corrigée
                )

                # Mettre à jour le DataArray avec la nouvelle transformation
                da.rio.write_transform(new_transform, inplace=True)

    # Définir les métadonnées par défaut
    default_metadata = {
        "title": da.name.replace("_", " ").title(),
        "description": f"Raster {da.name} computed from velocity cycle analysis.",
        "units": "1" if da.name.endswith(("idx", "doy")) else "m/year",
        "created_by": "Marie ZELLER",
        "institution": "IGE - UGA - CNRS",
        "date_created": str(np.datetime64("now")),
    }

    # Mettre à jour avec les métadonnées fournies
    if metadata:
        default_metadata.update(metadata)

    da.attrs.update(default_metadata)

    # Exporter en GeoTIFF
    da.rio.to_raster(
        output_path,
        dtype=dtype,
        nodata=nodata,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    print(f"Exported {output_path.name} with metadata and CRS EPSG:{epsg}.")

import rioxarray

def load_and_align_raster(path: Path, ref_grid: xr.DataArray) -> xr.DataArray:
    """Charge et aligne un raster GeoTIFF sur une grille de référence."""
    da = rioxarray.open_rasterio(path).squeeze()
    return da.rio.reproject_match(ref_grid)

def load_and_interp_geotiff_to_grid(geotiff_path, ref_dataarray, target_epsg="32632"):
    """Charge un GeoTIFF, l'interpole sur la grille d'un DataArray de référence et retourne un numpy array."""
    da = rioxarray.open_rasterio(geotiff_path).squeeze()
    if not da.rio.crs:
        da.rio.write_crs(target_epsg, inplace=True)
        # S'assurer que ref_dataarray a bien ses dimensions spatiales définies
    ref_dataarray = ref_dataarray.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    # Projeter da sur la grille de ref_dataarray
    da = da.rio.reproject_match(ref_dataarray.isel({dim: 0 for dim in ref_dataarray.dims if dim != 'mid_date'}))
    return da.values

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


