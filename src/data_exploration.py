from utils import *
import numpy as np
import xarray as xr
import rasterio
from pyproj import Transformer
from pathlib import Path

### ----- Background satellite image -----

tif_map = data_topo / "T32TLR_20250808T102701_TCI_60m.tif"
with rasterio.open(tif_map) as src:
    img_map = src.read([1,2,3])     # R,G,B
    bounds = src.bounds
    extent_map = [bounds.left, bounds.right, bounds.bottom, bounds.top]


### ----- DEM file -----

dem_file = data_topo / "Mt_Blanc_small_UTM32N.tif"


### ----- GPS data for validation -----

## WHEEL
x_l2_Argw, y_l2_Argw = 45.967186, 6.966498
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632")
x_utm_Argw, y_utm_Argw = transformer.transform(x_l2_Argw, y_l2_Argw)

data_Argwheel = pd.read_csv(data_validation / "wheel_SlidingVelocities1989_2019.csv", sep = ';', header=1, names = ['date', 'velocity'])
data_Argwheel['date'] = pd.to_datetime(data_Argwheel['date'])

data_Argwheel_end = pd.read_csv(data_validation / "cavitometer_2019-2021.dat", sep="\s+" , names = ["date", "velocity"], date_format="date")

data_Argwheel['velocity'] = pd.to_numeric(data_Argwheel['velocity'], errors='coerce')
data_Argwheel = data_Argwheel.dropna(subset=['date', 'velocity'])


## ARG 4

data_GPS_ARG4 = pd.read_csv(data_validation / "mean_kf_filled.vel", sep = '\s+', comment= '>', header=None, names = ['date', 'velocity'])
data_GPS_ARG4['date'] = pd.to_datetime(data_GPS_ARG4['date'])

x_lonlat_Arg_GPS1, y_lonlat_Arg_GPS1 = 45.9630388456039, 6.97477107923573
x_lonlat_Arg_GPS5, y_lonlat_Arg_GPS5 = 45.9647753102041, 6.97165435223673

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632")
x_utm_Arg_GPS1, y_utm_Arg_GPS1 = transformer.transform(x_lonlat_Arg_GPS1, y_lonlat_Arg_GPS1)
x_utm_Arg_GPS5, y_utm_Arg_GPS5 = transformer.transform(x_lonlat_Arg_GPS5, y_lonlat_Arg_GPS5)
x_utm_Arg4_GPS, y_utm_Arg4_GPS = 1/2 * (x_utm_Arg_GPS1 + x_utm_Arg_GPS5), 1/2 * (y_utm_Arg_GPS1 + y_utm_Arg_GPS5)

data_GPS_ARG4['velocity'] = pd.to_numeric(data_GPS_ARG4['velocity'], errors='coerce')
data_GPS_ARG4 = data_GPS_ARG4.dropna(subset=['date', 'velocity'])

## ARG G

data_GPS_ARGG = pd.read_csv(data_validation / "argg_2019-21.vel", sep = '\s+', comment= '>', header=None)
data_GPS_ARGG = data_GPS_ARGG[~data_GPS_ARGG[0].astype(str).str.startswith('#')] # Supprimer manuellement les lignes texte qui commencent par '#'
data_GPS_ARGG.columns = ['date', 'velocity']
data_GPS_ARGG['date'] = pd.to_datetime(data_GPS_ARGG['date'])

x_l2_ArgG_GPS, y_l2_ArgG_GPS = 961363.43, 2115529.11

transformer = Transformer.from_crs("EPSG:27572", "EPSG:32632")
x_utm_ArgG_GPS, y_utm_ArgG_GPS = transformer.transform(x_l2_ArgG_GPS, y_l2_ArgG_GPS)






### ----- Main dataset -----

# file_list = ["../archive/data/Cubes_TICOI/c_x00980_y03920_interp.nc", "../archive/data/Cubes_TICOI/c_x01225_y03675_interp.nc", 
#              "../archive/data/Cubes_TICOI/c_x01225_y03920_interp.nc", "../archive/data/Cubes_TICOI/c_x01470_y03430_1_interp.nc", 
#              "../archive/data/Cubes_TICOI/c_x01470_y03430_2_interp.nc", "../archive/data/Cubes_TICOI/c_x01470_y03675_1_interp.nc", 
#              "../archive/data/Cubes_TICOI/c_x01470_y03675_2_interp.nc", "../archive/data/Cubes_TICOI/c_x01715_y03430_interp.nc"] 

# ds_list = [xr.open_dataset(f) for f in file_list]

# merge_datasets(ds_list).to_netcdf("../data/ds_merged.nc")

ds_merged = xr.open_dataset(data_dir / "ds_merged.nc")

dx = float(ds_merged['x'][1] - ds_merged['x'][0])
dy = float(ds_merged['y'][1] - ds_merged['y'][0])
pixel_size = (dx + dy) / 2

ds_merged = ds_merged.assign_coords(dayofyear=ds_merged['mid_date'].dt.dayofyear.data, year=ds_merged['mid_date'].dt.year.data)

velocity = np.sqrt(ds_merged['vx']**2 + ds_merged['vy']**2)
xcount = np.sqrt(ds_merged['xcount_x']**2 + ds_merged['xcount_y']**2)

# Skip incomplete years
ds_merged = ds_merged.where((ds_merged['mid_date'].dt.year >= 2016) & (ds_merged['mid_date'].dt.year <= 2022), drop=True)

