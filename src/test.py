from utils import *
from data_exploration import *
from peaks import *
from masks import *
from melt import *
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

file_path = out_dir / "analysis_dataset.nc"

# Charger le Dataset
ds_analysis = xr.open_dataset(file_path)
vel_result = ds_analysis["velocity"]
vel_cycle = ds_analysis["vel_cycle"]
max_peak_doy = ds_analysis["max_peak_doy"]
amplitude_rel = ds_analysis["amplitude_rel"]
mask_stable_areas = ds_analysis["mask_stable_areas"] # & base_mask


out_dir_without_spatial_avg = script_dir / ".." / "data" / "output_without_spatial_avg"
file_path2 = out_dir_without_spatial_avg / "analysis_dataset.nc"

ds_analysis2 = xr.open_dataset(file_path2)
vel_result2 = ds_analysis2["velocity"]
vel_cycle2 = ds_analysis2["vel_cycle"]
max_peak_doy2 = ds_analysis2["max_peak_doy"]
amplitude_rel2 = ds_analysis2["amplitude_rel"]
mask_stable_areas2 = ds_analysis2["mask_stable_areas"] # & base_mask


import matplotlib.pyplot as plt
import numpy as np

# --- Masques ---
mask1 = mask_stable_areas & (max_peak_doy != 0)
mask2 = mask_stable_areas2 & (max_peak_doy2 != 0)

amp1 = amplitude_rel.where(mask1)
phase1 = max_peak_doy.where(mask1)

amp2 = amplitude_rel2.where(mask2)
phase2 = max_peak_doy2.where(mask2)

print("amp1 : ", np.nanmean(amp1), np.nanmedian(amp1), np.nanstd(amp1), "phase1 : ", np.nanmean(phase1), np.nanmedian(phase1), np.nanstd(phase1))
print("amp2 : ", np.nanmean(amp2), np.nanmedian(amp2), np.nanstd(amp2), "phase2 : ", np.nanmean(phase2), np.nanmedian(phase2), np.nanstd(phase2))


# --- Figures ---
fig, axs = plt.subplots(2, 2, figsize=(12,10))

im0 = axs[0,0].imshow(amp1, cmap="viridis")
axs[0,0].set_title("Amplitude (spatial avg)")
plt.colorbar(im0, ax=axs[0,0], fraction=0.046)

im1 = axs[0,1].imshow(phase1, cmap="twilight")
axs[0,1].set_title("Phase / peak DOY (spatial avg)")
plt.colorbar(im1, ax=axs[0,1], fraction=0.046)

im2 = axs[1,0].imshow(amp2, cmap="viridis")
axs[1,0].set_title("Amplitude (no spatial avg)")
plt.colorbar(im2, ax=axs[1,0], fraction=0.046)

im3 = axs[1,1].imshow(phase2, cmap="twilight")
axs[1,1].set_title("Phase / peak DOY (no spatial avg)")
plt.colorbar(im3, ax=axs[1,1], fraction=0.046)

for ax in axs.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# --- Définir les chemins de sortie ---
out_amp = out_dir / "amplitude_rel_stable_areas.tif"
out_phase = out_dir / "phase_peak_doy_stable_areas.tif"

out_amp2 = out_dir_without_spatial_avg / "amplitude_rel_stable_areas_no_spatial_avg.tif"
out_phase2 = out_dir_without_spatial_avg / "phase_peak_doy_stable_areas_no_spatial_avg.tif"

# --- Écriture des GeoTIFF ---
amp1.rio.to_raster(out_amp)
phase1.rio.to_raster(out_phase)

amp2.rio.to_raster(out_amp2)
phase2.rio.to_raster(out_phase2)





# # Loading raw data
# print("Loading raw data...")
# velocity = np.sqrt(ds_merged['vx']**2 + ds_merged['vy']**2)
# xcount = np.sqrt(ds_merged['xcount_x']**2 + ds_merged['xcount_y']**2)

# # Processing velocity timeseries
# print("Processing velocity timeseries...")

# # # Spatial smoothing
# # vel_avg_spatial = xr.apply_ufunc(
# #     convolve_2d,
# #     velocity,
# #     kwargs={'kernel': kernel},
# #     input_core_dims=[['y', 'x']],
# #     output_core_dims=[['y', 'x']],
# #     vectorize=True,
# #     dask='parallelized' if velocity.chunks else False,
# #     output_dtypes=[velocity.dtype]
# # )

# vel_avg_spatial = velocity

# # Detrending timeseries
# vel_avg_detrended = xr.apply_ufunc(
#     detrend_1d_brutal,
#     vel_avg_spatial,
#     input_core_dims=[['mid_date']],
#     output_core_dims=[['mid_date']],
#     vectorize=True,
#     dask='parallelized' if vel_avg_spatial.chunks else False,
#     output_dtypes=[vel_avg_spatial.dtype]
# ).rename("vel_avg_detrended")

# vel_avg_detrended = vel_avg_detrended.transpose('mid_date', 'y', 'x')

# print(vel_avg_spatial.shape)
# print(vel_avg_detrended.shape)

# # Selectiong final velocity timeseries
# vel_result = vel_avg_detrended

# # Computing mean annual cycle
# print("Computing average annual cycle...")

# doys = vel_result['mid_date'].dt.dayofyear.values
# cycle_array = create_avg_year_smooth(vel_result, doys)


# velocity_cycle_mean = xr.DataArray(
#     cycle_array,
#     dims=('cycle', 'y', 'x'),
#     coords={'cycle': np.arange(1, 366), 'y': vel_result['y'], 'x': vel_result['x']},
#     name="velocity_cycle_mean"
# ).assign_coords(doy_approx=("cycle", np.arange(1, 366)))


# velocity_masked = velocity.where(mask_stable_areas)


# # velocity_2d = velocity.isel(mid_date=0)
# # plt.figure(figsize=(10,6))
# # velocity_2d.plot(cmap="viridis")
# # plt.title("Velocity")
# # plt.show()

# # velocity_masked_2d = velocity_masked.isel(mid_date=0)
# # plt.figure(figsize=(10,6))
# # velocity_masked_2d.plot(cmap="viridis")
# # plt.title("Velocity in stable areas")
# # plt.show()

# print(max_peak_doy.dims)

# print("Shape mask_stable_areas :", mask_stable_areas.shape)
# print("Shape max_peak_doy :", max_peak_doy.shape)

# # Combien de pixels stables
# print("Nb pixels dans mask stable :", mask_stable_areas.sum().item())

# # Nb pixels velocity valides
# mask_valid_velocity = mask_stable_areas & velocity.notnull().any(dim='mid_date')
# print("Nb pixels stables avec velocity values :", mask_valid_velocity.sum().item())


# max_peak_doy_stable = max_peak_doy.where(mask_stable_areas)

# n_nonzero = ((max_peak_doy_stable.notnull()) & (max_peak_doy_stable != 0)).sum().item()
# print("Nombre de pixels DOY ≠ 0 :", n_nonzero)

# # plt.figure(figsize=(8,6))
# # max_peak_doy_stable.plot(cmap="hsv")  # ou "gray", "plasma", etc.
# # plt.title("max_peak_doy sur zones stables")
# # plt.show()


# mask_valid = (mask_stable_areas) & (velocity.notnull().any(dim='mid_date')) & (max_peak_doy == 0)

# # Indices des pixels qui respectent cette condition
# ys, xs = np.where(mask_valid)
# print(f"Nombre de pixels stables avec velocity valide mais max_peak_doy = 0 : {len(ys)}")

# import random

# n_samples = 1  # nombre de pixels à tracer
# selected_idx = random.sample(range(len(ys)), n_samples)

# pixels_to_plot = [(ys[i], xs[i]) for i in selected_idx]
# print("Pixels sélectionnés pour le plot :", pixels_to_plot)

# for y, x in pixels_to_plot:

#     print("velocity dims:", velocity.dims)
#     print("vel_result dims:", vel_result.dims)

#     print("Série temporelle brute :")
#     print(velocity[:, y, x].values)

#     print("Nombre de NaN dans la série brute :",
#         np.isnan(velocity[:, y, x].values).sum())
    
#     print("Série temporelle moyennée :")
#     print(vel_avg_spatial[:, y, x].values)

#     print("Nombre de NaN dans la série moyennée :",
#         np.isnan(vel_avg_spatial[:, y, x].values).sum())
    
#     print("Série temporelle moyennée détrendée :")
#     print(vel_result[:, y, x].values)

#     print("Nombre de NaN dans la série moyennée détrendée :",
#         np.isnan(vel_result[:, y, x].values).sum())

#     print("Cycle annuel convolué pour ce pixel :")
#     print(cycle_array[:, y, x])

#     print("Nombre de NaN dans le cycle :",
#         np.isnan(cycle_array[:, y, x]).sum())

#     print("NaN dans velocity_cycle_mean :",
#         np.isnan(velocity_cycle_mean[:, y, x]).sum().item())
        
# plt.figure(figsize=(12,6))

# for y, x in pixels_to_plot:
#     # extraire la série temporelle pour ce pixel
#     ts = velocity[:, y, x]  # shape = (time,)
#     plt.plot(ds_merged['mid_date'], ts, label=f"Pixel ({y},{x})")

# plt.xlabel("Time")
# plt.ylabel("Velocity (m/year)")
# plt.title("Séries temporelles des pixels stables")
# plt.legend()
# plt.show()

# for y, x in pixels_to_plot:
#     # extraire la série temporelle pour ce pixel
#     ts = vel_cycle[:, y, x]  # shape = (time,)
#     plt.plot(vel_cycle['doy_approx'], ts, label=f"Pixel ({y},{x})")

# plt.xlabel("Time")
# plt.ylabel("Velocity (m/year)")
# plt.title("Séries temporelles des pixels stables")
# plt.legend()
# plt.show()


# max_peak_doy_stable.rio.to_raster(
#     out_dir / "max_peak_doy_stable.tif",
#     compress="LZW"
# )

# # import rasterio


# # # Ouvrir le raster de référence avec rasterio
# # with rasterio.open(out_dir / "max_peak_doy.tif") as src:
# #     print("Bounds:", src.bounds)  # Vérifier les bornes géographiques
# #     print("Transform:", src.transform)  # Vérifier la transformation
# #     print("Width:", src.width)  # Largeur du raster
# #     print("Height:", src.height)  # Hauteur du raster
# #     print("CRS:", src.crs)  # Vérifier le CRS



# # # Vérifier les propriétés de max_peak_doy.tif
# # with rasterio.open(out_dir / "max_peak_doy.tif") as src:
# #     print("Transformation de max_peak_doy.tif :", src.transform)
# #     print("CRS de max_peak_doy.tif :", src.crs)
# #     print("Bornes de max_peak_doy.tif :", src.bounds)


