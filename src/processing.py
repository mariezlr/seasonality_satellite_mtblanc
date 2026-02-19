from utils import *
from data_exploration import *
from peaks import *
from masks import *
from melt import *
import numpy as np
import xarray as xr

# Loading raw data
print("Loading raw data...")
velocity = np.sqrt(ds_merged['vx']**2 + ds_merged['vy']**2)
xcount = np.sqrt(ds_merged['xcount_x']**2 + ds_merged['xcount_y']**2)

# Processing velocity timeseries
print("Processing velocity timeseries...")

# Spatial smoothing
vel_avg_spatial = xr.apply_ufunc(
    convolve_2d,
    velocity,
    kwargs={'kernel': kernel},
    input_core_dims=[['y', 'x']],
    output_core_dims=[['y', 'x']],
    vectorize=True,
    dask='parallelized' if velocity.chunks else False,
    output_dtypes=[velocity.dtype]
)

# Detrending timeseries
vel_avg_detrended = xr.apply_ufunc(
    detrend_1d_brutal,
    vel_avg_spatial,
    input_core_dims=[['mid_date']],
    output_core_dims=[['mid_date']],
    vectorize=True,
    dask='parallelized' if vel_avg_spatial.chunks else False,
    output_dtypes=[vel_avg_spatial.dtype]
).rename("vel_avg_detrended")

# NaNs Interpolation 
vel_avg_interp = xr.apply_ufunc(
    interp_1d_fill,
    vel_avg_detrended,
    ds_merged['mid_date'],
    input_core_dims=[['mid_date'], ['mid_date']],
    output_core_dims=[['mid_date']],
    vectorize=True,
    dask='parallelized' if vel_avg_detrended.chunks else False,
    output_dtypes=[vel_avg_detrended.dtype]
)

# Lowpass fft filtering
vel_lowpass = xr.apply_ufunc(
    fft_filter,
    vel_avg_interp,
    input_core_dims=[['mid_date']],
    output_core_dims=[['mid_date']],
    vectorize=True,
    dask='parallelized' if vel_avg_interp.chunks else False,
    output_dtypes=[vel_avg_interp.dtype]
).rename("vel_lowpass")

# Selectiong final velocity timeseries
vel_result = vel_lowpass if result_lowpass else vel_avg_detrended

# Computing mean annual cycle
print("Computing average annual cycle...")

doys = vel_result['mid_date'].dt.dayofyear.values
cycle_array = create_avg_year_smooth(vel_result, doys)

velocity_cycle_mean = xr.DataArray(
    cycle_array,
    dims=('cycle', 'y', 'x'),
    coords={'cycle': np.arange(1, 366), 'y': vel_result['y'], 'x': vel_result['x']},
    name="velocity_cycle_mean"
).assign_coords(doy_approx=("cycle", np.arange(1, 366)))

# Computing peaks and inflexion points
print("Computing peaks and inflexion points...")

peaks_ds = compute_peaks_and_export(velocity_cycle_mean, out_dir=out_dir, epsg=32632)
inflex_ds = compute_inflex_and_export(velocity_cycle_mean, out_dir=out_dir, epsg=32632)

amplitude = (peaks_ds["max_peak_vals"] - peaks_ds["min_peak_vals"])
avg_velocity = velocity_cycle_mean.mean(dim="cycle")
amplitude_rel = (peaks_ds["max_peak_vals"] - peaks_ds["min_peak_vals"]) / avg_velocity

# Computing masks
print("Computing masks...")

shadow_raster_path = mask_dir / "shadow_map_border_merged.tif"
mask_total, base_mask, mask_xcount, mask_shadow, mask_velavg, mask_snr = compute_total_mask_and_export(
    output_dir=mask_dir,
    xcount=xcount,
    vel_result=vel_result,
    shadow_raster_path=shadow_raster_path,
    min_valid_obs=80,
    threshold_xcount=100,
    threshold_shadow=50,
    threshold_velavg=20,
    snr_threshold=10,
)

stable_areas_geospatial_path = mask_dir / "stable_areas" / "crop_stable_areas_massif_without_mask_Diego.gpkg"
mask_stable_areas = compute_mask_stable_areas_and_export(output_dir=mask_dir, vel_result=vel_result, stable_areas_geospatial_path = stable_areas_geospatial_path)

mask_valid = mask_total
# mask_valid = base_mask & mask_xcount & mask_shadow & mask_velavg & mask_snr

# Chargement et alignement des rasters

print("Loading and aligning masks and peak timing rasters...")

# Définir la grille de référence
ref_grid = vel_result.isel(mid_date=0)
ref_grid = ref_grid.rio.write_crs("EPSG:32632", inplace=False)
ref_grid = ref_grid.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

# Corriger la transformation de la grille de référence

# Charger et aligner les masques
mask_total = load_and_align_raster(mask_dir / "mask_total.tif", ref_grid).astype(bool)
mask_xcount = load_and_align_raster(mask_dir / "mask_xcount.tif", ref_grid).astype(bool)
mask_shadow = load_and_align_raster(mask_dir / "mask_shadow.tif", ref_grid).astype(bool)
mask_velavg = load_and_align_raster(mask_dir / "mask_velavg.tif", ref_grid).astype(bool)
mask_snr = load_and_align_raster(mask_dir / "mask_snr.tif", ref_grid).astype(bool)
base_mask = load_and_align_raster(mask_dir / "base_mask.tif", ref_grid).astype(bool)

mask_stable_areas = load_and_align_raster(mask_dir / "mask_stable_areas.tif", ref_grid).astype(bool)

# Charger et aligner les rasters de timing des pics
max_peak_doy = load_and_align_raster(out_dir / "max_peak_doy.tif", ref_grid)
min_peak_doy = load_and_align_raster(out_dir / "min_peak_doy.tif", ref_grid)
inflex_doy = load_and_align_raster(out_dir / "inflex_doy.tif", ref_grid)

# Processing DEM and computing slope
print("Processing DEM and computing slope...")

dem_interp = load_and_align_raster(dem_file, ref_grid)

dem_da = xr.DataArray(
    dem_interp,
    coords={"y": ref_grid.y, "x": ref_grid.x},
    dims=["y", "x"],
    name="elevation"
)

dx = ref_grid.x.values[1] - ref_grid.x.values[0]
dy = ref_grid.y.values[1] - ref_grid.y.values[0]
print(f"dx: {dx}, dy: {dy}")

slope_interp = compute_slope(dem_interp, dx, dy)

# Afficher les statistiques de slope_interp
print("Statistiques de slope_interp :")
print(f"Min: {np.nanmin(slope_interp)}, Max: {np.nanmax(slope_interp)}")
print(f"Moyenne: {np.nanmean(slope_interp)}, Écart-type: {np.nanstd(slope_interp)}")

slope_da = xr.DataArray(
    slope_interp,
    coords={"y": ref_grid.y, "x": ref_grid.x},
    dims=["y", "x"],
    name="slope"
)

# Loading and aligning melt cycle
print("Loading and aligning melt cycle...")

melt_da_path = Path(out_dir / "melt_cycle_da.nc")

if not melt_da_path.exists():
    print(f"File {melt_da_path} doesn't exist. Creation of the file...")
    
    create_melt_cycle_da(melt_da_path, melt_file, dem_da, ref_grid)

melt_cycle_ds = xr.open_dataset(melt_da_path)
melt_cycle_da = melt_cycle_ds["melt_cycle_mean"]
melt_cycle_da = melt_cycle_da.transpose('cycle', 'y', 'x')

if not melt_cycle_da.rio.crs:
    melt_cycle_da = melt_cycle_da.rio.write_crs(ref_grid.rio.crs)

melt_cycle_da = melt_cycle_da.rio.reproject_match(ref_grid)

# Applying global mask
print("Applying global mask...")

vel_masked = vel_result.where(mask_valid)
slope_masked = slope_da.where(mask_valid)
dem_masked = dem_da.where(mask_valid)
amplitude_masked = amplitude.where(mask_valid)
amplitude_rel_masked = amplitude_rel.where(mask_valid)
avg_velocity_masked = avg_velocity.where(mask_valid)
melt_cycle_masked = melt_cycle_da.where(mask_valid)
max_peak_doy_masked = max_peak_doy.where(mask_valid)
min_peak_doy_masked = min_peak_doy.where(mask_valid)
inflex_doy_masked = inflex_doy.where(mask_valid)

# Creating and saving final dataset
print("Creating and saving final dataset...")

ds_analysis = xr.Dataset(
    {
        "velocity": vel_result,
        "vel_detrended": vel_avg_detrended,
        "vel_lowpass": vel_lowpass,
        "vel_cycle": velocity_cycle_mean,
        "avg_velocity": avg_velocity,
        "amplitude": amplitude,
        "amplitude_rel": amplitude_rel,
        "melt_cycle": melt_cycle_da,
        "slope": slope_da,
        "elevation": dem_da,
        "max_peak_doy": max_peak_doy,
        "min_peak_doy": min_peak_doy,
        "inflex_doy": inflex_doy,
        "mask": mask_valid,
        "mask_xcount" : mask_xcount,
        "mask_shadow" : mask_shadow,
        "mask_velavg" : mask_velavg,
        "mask_snr" : mask_snr,
        "base_mask" : base_mask,
        "mask_stable_areas" : mask_stable_areas

    }
)

# Adding global metadata
ds_analysis.attrs = {
    "title": "Mont Blanc Seasonality Analysis Dataset",
    "description": "Dataset containing velocity, slope, masks, and melt cycle data.",
    "author": "Marie ZELLER",
    "institution": "IGE - UGA - CNRS",
    "date_created": str(np.datetime64("now")),
    "crs": "EPSG:32632",
}

# Saving final dataset
ds_analysis.to_netcdf(out_dir / "analysis_dataset.nc")
print("Final dataset saved to:", out_dir / "analysis_dataset.nc")
