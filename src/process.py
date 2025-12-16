from utils import *
from data_exploration import *
from peaks import *
from masks import *
import numpy as np
import xarray as xr
import xdem
import geoutils as gu


# Velocity timeseries
velocity = np.sqrt(ds_merged['vx']**2 + ds_merged['vy']**2)
xcount = np.sqrt(ds_merged['xcount_x']**2 + ds_merged['xcount_y']**2)

# Spatial avg on dimension 'mid_date'
vel_avg_spatial = xr.apply_ufunc(
    convolve_2d,
    velocity,
    kwargs={'kernel': kernel},
    input_core_dims=[['y', 'x']],
    output_core_dims=[['y', 'x']],
    vectorize=True,
    dask='parallelized' if velocity.chunks else False,
    output_dtypes=[velocity.dtype])

print("Spatial 3×3 average computed")

# For each point, detrend velocity timeseries
vel_avg_detrended = xr.apply_ufunc(
    detrend_1d,
    vel_avg_spatial,
    input_core_dims=[['mid_date']],
    output_core_dims=[['mid_date']],
    vectorize=True,
    dask='parallelized' if vel_avg_spatial.chunks else False,
    output_dtypes=[vel_avg_spatial.dtype])

print("Timeseries detrended")

vel_result = vel_avg_detrended

# Interpolation to fill NaNs before fft
vel_avg_interp = xr.apply_ufunc(
    interp_1d_fill,
    vel_avg_detrended,
    ds_merged['mid_date'],
    input_core_dims=[['mid_date'], ['mid_date']],
    output_core_dims=[['mid_date']],
    vectorize=True,
    dask='parallelized' if vel_avg_detrended.chunks else False,
    output_dtypes=[vel_avg_detrended.dtype])
print("NaNs filled by interpolation")

# FFT lowpass filter for each pixel
vel_lowpass = xr.apply_ufunc(
    fft_filter,
    vel_avg_interp,
    input_core_dims=[['mid_date']],
    output_core_dims=[['mid_date']],
    vectorize=True,
    dask='parallelized' if vel_avg_interp.chunks else False,
    output_dtypes=[vel_avg_interp.dtype])
print("FFT low-pass filter applied")

if result_lowpass:
    vel_result = vel_lowpass


# Average annual cycle
doys = vel_result['mid_date'].dt.dayofyear.values
cycle_array = create_avg_year_smooth(vel_result, doys)

velocity_cycle_mean = xr.DataArray(
    cycle_array,  # ndarray lissé
    dims=('cycle','y','x'),
    coords={'cycle': np.arange(1,366), 'y': vel_result['y'], 'x': vel_result['x']})

velocity_cycle_mean = velocity_cycle_mean.assign_coords(
    doy_approx = velocity_cycle_mean['cycle'])

print("Average annual cycle computed")



# Save 3D arrays as NetCDF
print("Exporting velocities as NetCDF files...")
velocity_cycle_mean.to_netcdf(out_dir / "velocity_cycle_mean.nc")
vel_avg_detrended.to_netcdf(out_dir / "vel_avg_detrended.nc")
vel_lowpass.to_netcdf(out_dir / "vel_lowpass.nc")

print("Computing peaks + exporting GeoTIFF…")
peaks_ds = compute_peaks_and_export(velocity_cycle_mean, out_dir=out_dir, epsg=32632)
inflex_ds = compute_inflex_and_export(velocity_cycle_mean, out_dir=out_dir, epsg=32632)

# Masks
mask_total_path = mask_dir / "mask_total.tif"
shadow_raster_path = mask_dir / "shadow_map_border_merged.tif"

print("Computing masks + exporting GeoTIFF…")
mask_total, base_mask, mask_xcount, mask_shadow, mask_snr = compute_total_mask_and_export(
    mask_dir, xcount, vel_result, shadow_raster_path, 
    min_valid_obs=80, threshold_xcount=100, threshold_shadow=50, snr_threshold=5)

print("base_mask : ", mask_stats(base_mask, vel_result))
print("mask_xcount : ", mask_stats(mask_xcount, vel_result))
print("mask_shadow : ", (mask_shadow, vel_result))
print("mask_snr : ", mask_stats(mask_snr, vel_result))
print("mask_total : ", mask_stats(mask_total, vel_result))

# Melt rate
result = gu.Raster(out_dir / "max_peak_doy.tif")

dem = xdem.DEM(dem_file) 
dem_reproj = dem.reproject(result)

print("Computing melt rate + exporting NetCDF...")
dem_da = xr.DataArray(dem_reproj.data, dims=["y", "x"],
                      coords={"y": ds_merged.y, "x": ds_merged.x}, name="dem")

dates_melt = melt_file['date'].values
temps_melt = melt_file['temp'].values


# Create cycles : here we want cycle = dayofyear // 5 (0..)
doys_melt = pd.DatetimeIndex(dates_melt).dayofyear
cycles_melt = (doys_melt // 5).astype(int)

unique_cycles = np.unique(cycles_melt)

melt_cycle_mean, cycle_list = build_melt_cycle(dem_da, unique_cycles, dates_melt, temps_melt, cycles_melt)

# Convert to xarray DataArray with coord doy_approx = cycle*5
doy_approx_melt = np.array([c*5 for c in cycle_list])
melt_cycle_da = xr.DataArray(melt_cycle_mean,
   coords={"cycle": cycle_list, "y": dem_da.y, "x": dem_da.x, "doy_approx": ("cycle", doy_approx_melt)},
   dims=("cycle","y","x"))

melt_cycle_da.name = "melt_cycle_mean"
encoding = {"melt_cycle_mean": {"dtype": "float32", "zlib": True, "complevel": 4,}}
melt_cycle_da.to_netcdf(out_dir / "melt_cycle_mean.nc", engine="netcdf4", encoding=encoding)


