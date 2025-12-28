from utils import *
from data_exploration import *

import geoutils as gu

import rasterio

# Ouvrir le raster de référence avec rasterio
with rasterio.open(out_dir / "max_peak_doy.tif") as src:
    print("Bounds:", src.bounds)  # Vérifier les bornes géographiques
    print("Transform:", src.transform)  # Vérifier la transformation
    print("Width:", src.width)  # Largeur du raster
    print("Height:", src.height)  # Hauteur du raster
    print("CRS:", src.crs)  # Vérifier le CRS




import rasterio

# Vérifier les propriétés de max_peak_doy.tif
with rasterio.open(out_dir / "max_peak_doy.tif") as src:
    print("Transformation de max_peak_doy.tif :", src.transform)
    print("CRS de max_peak_doy.tif :", src.crs)
    print("Bornes de max_peak_doy.tif :", src.bounds)


