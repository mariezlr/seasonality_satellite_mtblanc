from utils import *
from data_exploration import *
import numpy as np
from scipy.ndimage import convolve
import xarray as xr

# velocity timeseries
velocity = np.sqrt(ds_merged['vx']**2 + ds_merged['vy']**2)

kernel = np.ones((3, 3), dtype=float) # uniform 3x3 kernel
kernel /= kernel.sum()  # normalize to get the avergae

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

print("Successfully calculated velocity spatial average !")

# For each point, detrend velocity timeseries
vel_avg_detrended = xr.apply_ufunc(
    detrend_1d,
    vel_avg_spatial,
    input_core_dims=[['mid_date']],
    output_core_dims=[['mid_date']],
    vectorize=True,
    dask='parallelized' if vel_avg_spatial.chunks else False,
    output_dtypes=[vel_avg_spatial.dtype])

print("Successfully calculated detrended velocity timeseries !")

# Interpolation to fill NaNs before fft
vel_avg_detrend_interp = xr.apply_ufunc(
    interp_1d_fill,
    vel_avg_detrended,
    ds_merged['mid_date'],
    input_core_dims=[['mid_date'], ['mid_date']],
    output_core_dims=[['mid_date']],
    vectorize=True,
    dask='parallelized' if vel_avg_detrended.chunks else False,
    output_dtypes=[vel_avg_detrended.dtype])

print("Successfully filled NaNs in velocity timeseries !")

# FFT lowpass filter for each pixel
vel_lowpass_filter = xr.apply_ufunc(
    fft_filter,
    vel_avg_detrend_interp,
    input_core_dims=[['mid_date']],
    output_core_dims=[['mid_date']],
    vectorize=True,
    dask='parallelized' if vel_avg_detrend_interp.chunks else False,
    output_dtypes=[vel_avg_detrend_interp.dtype])

print("Successfully calculated lowpass filter on velocity timeseries !")

vel_result = vel_avg_detrended

ds_merged['velocity_avg'] = vel_result

vel_result['dayofyear'] = vel_result['mid_date'].dt.dayofyear
