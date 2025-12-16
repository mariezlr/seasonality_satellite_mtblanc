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

# Load 3D velocity arrays (NetCDF)
print("Loading 3D velocity datasets...")
vel_avg_detrended = xr.open_dataarray(out_dir / "vel_avg_detrended.nc")
vel_lowpass = xr.open_dataarray(out_dir / "vel_lowpass.nc")
velocity_cycle_mean = xr.open_dataarray(out_dir / "velocity_cycle_mean.nc")

vel_result = vel_avg_detrended
if result_lowpass:
    vel_result = vel_lowpass

doys = vel_result['mid_date'].dt.dayofyear.values
cycle_array = create_avg_year_smooth(vel_result, doys)

valid_vel_mask = ~velocity_cycle_mean.isnull().all(dim='cycle')
valid_indices = np.argwhere(valid_vel_mask.values)

# Load 2D masks (GeoTIFF)
print("Loading masks...")
mask_total = rioxarray.open_rasterio(mask_dir / "mask_total.tif").squeeze().astype(bool)
base_mask = rioxarray.open_rasterio(mask_dir / "base_mask.tif").squeeze().astype(bool) 
mask_xcount = rioxarray.open_rasterio(mask_dir / "mask_xcount.tif").squeeze().astype(bool)
mask_shadow = rioxarray.open_rasterio(mask_dir / "mask_shadow.tif").squeeze().astype(bool) 
mask_snr = rioxarray.open_rasterio(mask_dir / "mask_snr.tif").squeeze().astype(bool)


# Load 2D peaks files (GeoTIFF) with geoutils
max_doy_tif = out_dir / "max_peak_doy.tif" # .tif of the results (max position, most rapid month...)
min_doy_tif = out_dir / "min_peak_doy.tif"
inflex_doy_tif = out_dir / "inflex_doy.tif"

mask_file = mask_dir / "mask_total.tif" # .tif of binary mask

max_peak_doy = gu.Raster(max_doy_tif).data # 2D numpy array
min_peak_doy = gu.Raster(min_doy_tif).data # 2D numpy array
inflex_doy = gu.Raster(inflex_doy_tif).data # 2D numpy array

result = gu.Raster(max_doy_tif)
mask = gu.Raster(mask_file)

mask.set_nodata(0, update_array=False)
result_mask = mask.reproject(result) # Reprojection on the results coordinates (EPSG and transform)
mask_arr = result_mask.data > 0  # 2D numpy array


# Slope data
dem = xdem.DEM(dem_file) 
dem_reproj = dem.reproject(result)
dem_slope = dem_reproj.slope() # Calculates slopes []
slope_arr = dem_slope.data

from scipy.ndimage import uniform_filter
slope_arr = uniform_filter(slope_arr, size=3, mode='nearest')

# slope_xr = xr.DataArray(slope_arr, dims=("y", "x"),
#                         coords={"y": ds_merged["y"], "x": ds_merged["x"]}, name="slope")

# slope_avg_spatial = xr.apply_ufunc(
#     convolve_2d,
#     slope_xr,
#     kwargs={"kernel": kernel},
#     input_core_dims=[["y", "x"]],
#     output_core_dims=[["y", "x"]],
#     vectorize=True,
#     dask="parallelized" if slope_xr.chunks else False,
#     output_dtypes=[slope_xr.dtype],
# )

# print("Spatial 3×3 average computed for slope")


## Compute altitude and amplitude data

z = dem_reproj.data # 2D : (y, x)
avg_velocity = velocity_cycle_mean.mean(dim="cycle")

max_vals_tif = out_dir / "max_peak_vals.tif"
min_vals_tif = out_dir / "min_peak_vals.tif"
max_peak_vals = gu.Raster(max_vals_tif).data # 2D numpy array
min_peak_vals = gu.Raster(min_vals_tif).data # 2D numpy array

amplitude = max_peak_vals - min_peak_vals

# Load melt_rate file

ds_melt = xr.open_dataset(out_dir / "melt_cycle_mean.nc")

melt_cycle_da = (ds_melt["melt_cycle_mean"]
                 .transpose("cycle", "y", "x")
                 .assign_coords(doy_approx=("cycle", ds_melt["cycle"].values * 5)))






