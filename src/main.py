# =========================
# Imports
# =========================
from data_exploration import *
import numpy as np
import xarray as xr
import rioxarray
import xdem
from scipy.ndimage import uniform_filter

# Velocity timeseries
velocity = np.sqrt(ds_merged['vx']**2 + ds_merged['vy']**2)
xcount = np.sqrt(ds_merged['xcount_x']**2 + ds_merged['xcount_y']**2)

# =========================
# 1. Load velocity datasets (REFERENCE GRID)
# =========================
print("Loading velocity data...")

vel_avg_detrended = xr.open_dataarray(out_dir / "vel_avg_detrended.nc")
vel_lowpass = xr.open_dataarray(out_dir / "vel_lowpass.nc")

vel_result = vel_lowpass if result_lowpass else vel_avg_detrended

# Grille de référence (2D)
ref_grid = vel_result.isel(mid_date=0)
ref_grid = ref_grid.rio.write_crs("EPSG:32632", inplace=False)  # si pas déjà défini

# =========================
# 2. Load + align ALL 2D masks (GeoTIFF → xarray)
# =========================
print("Loading and aligning masks...")

def load_mask(path):
    da = rioxarray.open_rasterio(path).squeeze()
    da = da.rio.reproject_match(ref_grid)
    return da.astype(bool)

mask_total  = load_mask(mask_dir / "mask_total.tif")
mask_xcount = load_mask(mask_dir / "mask_xcount.tif")
mask_shadow = load_mask(mask_dir / "mask_shadow.tif")
mask_snr    = load_mask(mask_dir / "mask_snr.tif")

mask_valid = mask_total & mask_xcount & mask_shadow & mask_snr

# =========================
# 3. Load peak timing rasters (GeoTIFF → xarray)
# =========================
print("Loading peak timing rasters...")

def load_raster(path):
    da = rioxarray.open_rasterio(path).squeeze()
    return da.rio.reproject_match(ref_grid)

max_peak_doy = load_raster(out_dir / "max_peak_doy.tif")
min_peak_doy = load_raster(out_dir / "min_peak_doy.tif")
inflex_doy   = load_raster(out_dir / "inflex_doy.tif")

# =========================
# 4. DEM → slope → xarray (aligned ONCE)
# =========================
print("Computing slope...")


import xdem
import xarray as xr
from scipy.ndimage import uniform_filter
from pathlib import Path

# -------------------------
# 1. Charger DEM
# -------------------------


# Charger et interpoler DEM sur la grille de ds_merged
# dem_da = load_geotiff_as_da(dem_file, ds_merged['vx'], target_epsg=32632, name="elevation")

# # Calculer la pente
# slope_da = compute_slope_da(dem_da)


dem_interp = load_and_interp_geotiff_to_grid(dem_file, ds_merged, target_epsg="32632")

dx = ds_merged.x.values[1] - ds_merged.x.values[0]
dy = ds_merged.y.values[1] - ds_merged.y.values[0]

dem_da = xr.DataArray(
        dem_interp[::-1, :],  # remettre y dans le bon ordre
        coords={"y": ds_merged.y, "x": ds_merged.x},
        dims=["y", "x"],
        name="elevation"
    )

slope = compute_slope(dem_interp, dx, dy)

slope_da = xr.DataArray(
        slope,
        coords=dem_da.coords,
        dims=dem_da.dims,
        name="slope"
    )


# Masquer si besoin
elev_masked = dem_da.where(mask_valid)

slope_masked = slope_da.where(mask_valid)



# elevation_da = load_and_interp_geotiff_to_grid(dem_file, ds_merged, target_epsg="32632")

# dx = ds_merged.x.values[1] - ds_merged.x.values[0]
# dy = ds_merged.y.values[1] - ds_merged.y.values[0]

# slope_da = compute_slope(elevation_da, dx, dy)




# -------------------------
# 6. (Optionnel) Sauvegarde
# -------------------------
slope_da.rio.to_raster(out_dir / "slope_aligned.tif")
dem_da.rio.to_raster(out_dir / "elevation_aligned.tif")



# Derived quantities (xarray ONLY)
print("Derived fields...")


velocity_cycle_mean = xr.open_dataarray(out_dir / "velocity_cycle_mean.nc")
velocity_cycle_mean = velocity_cycle_mean.rio.reproject_match(ref_grid)

avg_velocity = velocity_cycle_mean.mean(dim="cycle")

amplitude = (
    load_raster(out_dir / "max_peak_vals.tif")
    - load_raster(out_dir / "min_peak_vals.tif")
)

ds_melt = xr.open_dataset(out_dir / "melt_cycle_mean.nc")

melt_cycle_da = (
    ds_melt["melt_cycle_mean"]
    .transpose("cycle", "y", "x")
    .assign_coords(doy_approx=("cycle", ds_melt["cycle"].values * 5))
)

if not melt_cycle_da.rio.crs:
    melt_cycle_da = melt_cycle_da.rio.write_crs(ref_grid.rio.crs)

# Reprojection / alignement spatial
melt_cycle_da = melt_cycle_da.rio.reproject_match(ref_grid)

# =========================
# 6. Apply master mask (ONE PLACE ONLY)
# =========================
print("Applying mask...")

vel_masked = vel_result.where(mask_valid)
slope_masked = slope_da.where(mask_valid)
amplitude_masked = amplitude.where(mask_valid)
avg_velocity_masked = avg_velocity.where(mask_valid)
melt_cycle_masked = melt_cycle_da.where(mask_valid)
# =========================
# 7. Save aligned analysis-ready dataset
# =========================
print("Saving aligned dataset...")

# Appliquer masque sur les pics
max_peak_doy_masked = max_peak_doy.where(mask_valid)
min_peak_doy_masked = min_peak_doy.where(mask_valid)
inflex_doy_masked   = inflex_doy.where(mask_valid)

# Créer le dataset final
ds_analysis = xr.Dataset(
    {
        "velocity": vel_masked,
        "vel_detrended": vel_avg_detrended.rio.reproject_match(ref_grid),
        "vel_lowpass": vel_lowpass.rio.reproject_match(ref_grid),
        "vel_cycle": velocity_cycle_mean,
        "avg_velocity": avg_velocity_masked,
        "amplitude": amplitude_masked,
        "melt_cycle": melt_cycle_masked,
        "slope": slope_masked,
        "elevation": elev_masked,
        "max_peak_doy": max_peak_doy_masked,
        "min_peak_doy": min_peak_doy_masked,
        "inflex_doy": inflex_doy_masked,
        "mask": mask_valid,
    }
)

# Écriture finale
ds_analysis.to_netcdf(out_dir / "analysis_dataset.nc")
print("Dataset saved to:", out_dir / "analysis_dataset.nc")
