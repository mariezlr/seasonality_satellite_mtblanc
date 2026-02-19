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
mask_stable_areas = ds_analysis["mask_stable_areas"] # & base_mask


velocity = np.sqrt(ds_merged['vx']**2 + ds_merged['vy']**2)

velocity_masked = velocity.where(mask_stable_areas)


# velocity_2d = velocity.isel(mid_date=0)
# plt.figure(figsize=(10,6))
# velocity_2d.plot(cmap="viridis")
# plt.title("Velocity")
# plt.show()

# velocity_masked_2d = velocity_masked.isel(mid_date=0)
# plt.figure(figsize=(10,6))
# velocity_masked_2d.plot(cmap="viridis")
# plt.title("Velocity in stable areas")
# plt.show()

print(max_peak_doy.dims)

print("Shape mask_stable_areas :", mask_stable_areas.shape)
print("Shape max_peak_doy :", max_peak_doy.shape)

# Combien de pixels stables
print("Nb pixels dans mask stable :", mask_stable_areas.sum().item())

# Nb pixels velocity valides
mask_valid_velocity = mask_stable_areas & velocity.notnull().any(dim='mid_date')
print("Nb pixels stables avec velocity values :", mask_valid_velocity.sum().item())


max_peak_doy_stable = max_peak_doy.where(mask_stable_areas)

n_nonzero = ((max_peak_doy_stable.notnull()) & (max_peak_doy_stable != 0)).sum().item()
print("Nombre de pixels DOY ≠ 0 :", n_nonzero)

# plt.figure(figsize=(8,6))
# max_peak_doy_stable.plot(cmap="hsv")  # ou "gray", "plasma", etc.
# plt.title("max_peak_doy sur zones stables")
# plt.show()



mask_valid = (mask_stable_areas) & (velocity.notnull().any(dim='mid_date')) & (max_peak_doy == 0)

# Indices des pixels qui respectent cette condition
ys, xs = np.where(mask_valid)
print(f"Nombre de pixels stables avec velocity valide mais max_peak_doy = 0 : {len(ys)}")

import random

n_samples = 5  # nombre de pixels à tracer
selected_idx = random.sample(range(len(ys)), n_samples)

pixels_to_plot = [(ys[i], xs[i]) for i in selected_idx]
print("Pixels sélectionnés pour le plot :", pixels_to_plot)

plt.figure(figsize=(12,6))

for y, x in pixels_to_plot:
    # extraire la série temporelle pour ce pixel
    ts = velocity[:, y, x]  # shape = (time,)
    plt.plot(ds_merged['mid_date'], ts, label=f"Pixel ({y},{x})")

plt.xlabel("Time")
plt.ylabel("Velocity (m/year)")
plt.title("Séries temporelles des pixels stables")
plt.legend()
plt.show()

for y, x in pixels_to_plot:
    # extraire la série temporelle pour ce pixel
    ts = vel_cycle[:, y, x]  # shape = (time,)
    plt.plot(vel_cycle['doy_approx'], ts, label=f"Pixel ({y},{x})")

plt.xlabel("Time")
plt.ylabel("Velocity (m/year)")
plt.title("Séries temporelles des pixels stables")
plt.legend()
plt.show()


max_peak_doy_stable.rio.to_raster(
    out_dir / "max_peak_doy_masked.tif",
    compress="LZW"
)

# import rasterio


# # Ouvrir le raster de référence avec rasterio
# with rasterio.open(out_dir / "max_peak_doy.tif") as src:
#     print("Bounds:", src.bounds)  # Vérifier les bornes géographiques
#     print("Transform:", src.transform)  # Vérifier la transformation
#     print("Width:", src.width)  # Largeur du raster
#     print("Height:", src.height)  # Hauteur du raster
#     print("CRS:", src.crs)  # Vérifier le CRS



# # Vérifier les propriétés de max_peak_doy.tif
# with rasterio.open(out_dir / "max_peak_doy.tif") as src:
#     print("Transformation de max_peak_doy.tif :", src.transform)
#     print("CRS de max_peak_doy.tif :", src.crs)
#     print("Bornes de max_peak_doy.tif :", src.bounds)


