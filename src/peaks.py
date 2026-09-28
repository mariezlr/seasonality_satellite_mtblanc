from utils import *
from data_exploration import *
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from rasterio.transform import Affine


def peak_and_index(arr_1d):
    """
    Returns [max_val, max_idx, min_val, min_idx] for a 1D signal.
    arr_1d : velocity time series (1D)
    """
    out = np.full(4, np.nan, dtype=np.float64)
    
    if np.all(np.isnan(arr_1d)):
        return out
    
    max_idx = np.nanargmax(arr_1d)
    min_idx = np.nanargmin(arr_1d)
    
    out[0] = arr_1d[max_idx]
    out[1] = max_idx
    out[2] = arr_1d[min_idx]
    out[3] = min_idx
    
    return out


def idx_to_doy(idx_da, doy_array):
    """
    Converts max/min indexes to DOY, NaN if invalid.
    """
    idx_vals = idx_da.values
    out = np.full_like(idx_vals, np.nan, dtype=float)

     # filter invalid indexes
    valid = (idx_vals >= 0) & (idx_vals < len(doy_array))
    out[valid] = doy_array[idx_vals[valid].astype(int)]
    return xr.DataArray(out, coords=idx_da.coords, dims=idx_da.dims, name=(idx_da.name or "idx") + "_doy")



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



def compute_peaks_and_export(
    velocity_cycle_mean: xr.DataArray,
    out_dir: Path,
    epsg: int = 32632,
) -> Dict[str, xr.DataArray]:
    """
    Calcule les pics (valeurs max/min + indices) et exporte les résultats en GeoTIFF.
    Calcule également les rasters DOY et les exporte.

    Args:
        velocity_cycle_mean: DataArray avec la dimension 'cycle'.
        out_dir: Dossier de sortie pour les fichiers GeoTIFF.
        epsg: Code EPSG pour le CRS.

    Returns:
        Dictionnaire des DataArray calculés.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calculer les pics et indices
    result = xr.apply_ufunc(
        peak_and_index,
        velocity_cycle_mean,
        input_core_dims=[["cycle"]],
        output_core_dims=[["stat"]],
        vectorize=True,
        output_dtypes=[float],
        output_sizes={"stat": 4},
    )

    # Organiser les résultats
    datasets = {
        "max_peak_vals": result.sel(stat=0).rename("max_peak_vals"),
        "max_peak_idx": result.sel(stat=1).astype(int).rename("max_peak_idx"),
        "min_peak_vals": result.sel(stat=2).rename("min_peak_vals"),
        "min_peak_idx": result.sel(stat=3).astype(int).rename("min_peak_idx"),
    }

    # Exporter les pics en GeoTIFF
    for name, da in datasets.items():
        metadata = {"description": f"Raster of {name} computed from velocity cycle analysis."}
        export_to_geotiff(da, out_dir / f"{name}.tif", epsg=epsg, metadata=metadata)

    # Calculer les DOY à partir des indices
    doy_array = velocity_cycle_mean["doy_approx"].values
    max_peak_doy = idx_to_doy(datasets["max_peak_idx"], doy_array)
    min_peak_doy = idx_to_doy(datasets["min_peak_idx"], doy_array)

    # # Créer des DataArrays pour les DOY
    # max_peak_doy_da = xr.DataArray(
    #     max_peak_doy,
    #     coords={"y": velocity_cycle_mean.y, "x": velocity_cycle_mean.x},
    #     dims=["y", "x"],
    #     name="max_peak_doy"
    # )
    max_peak_doy.rio.write_crs(f"EPSG:{epsg}", inplace=True)

    # min_peak_doy_da = xr.DataArray(
    #     min_peak_doy,
    #     coords={"y": velocity_cycle_mean.y, "x": velocity_cycle_mean.x},
    #     dims=["y", "x"],
    #     name="min_peak_doy"
    # )
    min_peak_doy.rio.write_crs(f"EPSG:{epsg}", inplace=True)

    # Exporter les DOY en GeoTIFF
    for name, da in {"max_peak_doy": max_peak_doy, "min_peak_doy": min_peak_doy}.items():
        metadata = {
            "units": "day of year",
            "description": f"Day of year corresponding to {name.replace('_doy', '')} indices."
        }
        export_to_geotiff(da, out_dir / f"{name}.tif", epsg=epsg, metadata=metadata, nodata=-9999, dtype="int16")

    # Ajouter les DOY au dictionnaire de retour
    datasets.update({"max_peak_doy": max_peak_doy, "min_peak_doy": min_peak_doy})

    return datasets


def compute_inflex_and_export(
    velocity_cycle_mean: xr.DataArray,
    out_dir: Path,
    epsg: int = 32632,
) -> Dict[str, xr.DataArray]:
    """
    Calcule les points d'inflexion et exporte les résultats en GeoTIFF.
    Calcule également les rasters DOY et les exporte.

    Args:
        velocity_cycle_mean: DataArray avec la dimension 'cycle'.
        out_dir: Dossier de sortie pour les fichiers GeoTIFF.
        epsg: Code EPSG pour le CRS.

    Returns:
        Dictionnaire des DataArray calculés.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calculer les points d'inflexion
    inflex_points = xr.apply_ufunc(
        index_of_pre_peak_derivative,
        velocity_cycle_mean,
        input_core_dims=[["cycle"]],
        output_core_dims=[["stat"]],
        vectorize=True,
        output_dtypes=[float],
        output_sizes={"stat": 2},
    )

    # Organiser les résultats
    datasets = {
        "inflex_vals": inflex_points.sel(stat=0).rename("inflex_vals"),
        "inflex_idx": inflex_points.sel(stat=1).astype(int).rename("inflex_idx"),
    }

    # Exporter les points d'inflexion en GeoTIFF
    for name, da in datasets.items():
        metadata = {"description": f"Raster of {name} computed from velocity cycle analysis."}
        export_to_geotiff(da, out_dir / f"{name}.tif", epsg=epsg, metadata=metadata)

    # Calculer les DOY à partir des indices
    doy_array = velocity_cycle_mean["doy_approx"].values
    inflex_doy = idx_to_doy(datasets["inflex_idx"], doy_array)

    # Créer un DataArray pour les DOY
    inflex_doy_da = xr.DataArray(
        inflex_doy,
        coords={"y": velocity_cycle_mean.y, "x": velocity_cycle_mean.x},
        dims=["y", "x"],
        name="inflex_doy"
    )
    inflex_doy_da.rio.write_crs(f"EPSG:{epsg}", inplace=True)

    # Exporter les DOY en GeoTIFF
    metadata = {
        "units": "day of year",
        "description": "Day of year corresponding to inflexion indices."
    }
    export_to_geotiff(inflex_doy_da, out_dir / "inflex_doy.tif", epsg=epsg, metadata=metadata, nodata=-9999, dtype="int16")

    # Ajouter les DOY au dictionnaire de retour
    datasets["inflex_doy"] = inflex_doy_da

    return datasets


count_short_before_max = 0
count_no_local_max = 0

def index_of_pre_peak_derivative(arr_1d: np.ndarray) -> np.ndarray:
    """
    Trouve l'index du dernier maximum local de la dérivée avant le pic de vitesse.

    Args:
        arr_1d: Tableau 1D de valeurs de vitesse.

    Returns:
        Tableau de taille 2 avec [valeur, index] ou NaN si non trouvé.
    """
    global count_short_before_max, count_no_local_max
    out = np.full(2, np.nan, dtype=float)

    if np.all(np.isnan(arr_1d)) or len(arr_1d) < 3:
        return out

    idx_max = np.nanargmax(arr_1d)

    if idx_max <= 2:
        count_short_before_max += 1
        return out

    deriv = np.diff(arr_1d[:idx_max])
    local_max = np.where((deriv[1:-1] > deriv[:-2]) & (deriv[1:-1] > deriv[2:]))[0] + 1

    if len(local_max) == 0:
        count_no_local_max += 1
        return out

    selected_idx = local_max[-1]
    out[0] = arr_1d[selected_idx]
    out[1] = selected_idx

    return out
