from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

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



def create_melt_cycle_da(melt_da_path, melt_file, dem_da, ref_grid):
    """
    Creation of melt_cycle_da.nc.

    Args:
        melt_da_path (Path): Path for saving the final file.
        melt_file (xarray.Dataset): File containing melt data.
        dem_da (xarray.DataArray): Elevation data.
        ref_grid (xarray.Dataset): Reference grid for coordinates.
    """

    
    dates_melt = melt_file['date'].values
    temps_melt = melt_file['temp'].values
    
    # Create cycles
    doys_melt = pd.DatetimeIndex(dates_melt).dayofyear
    cycles_melt = np.where(doys_melt == 366, 365, doys_melt)  # To avoid error due to doy=366
    unique_cycles = np.unique(cycles_melt)
    
    # Assume build_melt_cycle and export_to_netcdf_with_metadata are defined elsewhere
    melt_cycle_mean, cycle_list = build_melt_cycle(dem_da, unique_cycles, dates_melt, temps_melt, cycles_melt)
    
    # Convert to xarray DataArray
    melt_cycle_da = xr.DataArray(
        melt_cycle_mean,
        dims=('cycle', 'y', 'x'),
        coords={'cycle': np.arange(1, 366), "y": ref_grid.y, "x": ref_grid.x},
        name="melt_cycle_mean"
    ).assign_coords(doy_approx=("cycle", np.arange(1, 366)))
    
    # Export melt_cycle_da with metadata
    export_to_netcdf_with_metadata(
        da=melt_cycle_da,
        output_path=melt_da_path,
        title="Melt Cycle Mean",
        description="Mean melt cycle computed from temperature timeseries.",
        units="°C/day",
        crs="EPSG:32632",
        encoding={"melt_cycle_mean": {"dtype": "float32", "zlib": True, "complevel": 4}}
    )
    
    print(f"File {melt_da_path} was created.")


