from utils import *
import numpy as np
import xarray as xr
import pandas as pd
import rasterio
from pyproj import Transformer
from pathlib import Path


kernel = np.ones((3, 3), dtype=float) / 9.0 # uniform 3x3 kernel normalized to get the avergae


### ----- Variables -----

result_lowpass = False # False if we want to process the averaged detrended data rather than lowpass filtered data

slope_bins = np.arange(0, 45, 5)
n_bins = len(slope_bins) - 1
month_bins = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
mid_month_days = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
month_labels_short = ['J','F','M','A','M','J','J','A','S','O','N','D']
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

years = np.arange(2016, 2023)


### ----- Main dataset -----

ds_path = Path(data_dir / "ds_merged.nc")

if not ds_path.exists():
    print(f"File {ds_path} doesn't exist. Creation of the file...")

    file_list = ["../archive/data/Cubes_TICOI/c_x00980_y03920_interp.nc", "../archive/data/Cubes_TICOI/c_x01225_y03675_interp.nc", 
                "../archive/data/Cubes_TICOI/c_x01225_y03920_interp.nc", "../archive/data/Cubes_TICOI/c_x01470_y03430_1_interp.nc", 
                "../archive/data/Cubes_TICOI/c_x01470_y03430_2_interp.nc", "../archive/data/Cubes_TICOI/c_x01470_y03675_1_interp.nc", 
                "../archive/data/Cubes_TICOI/c_x01470_y03675_2_interp.nc", "../archive/data/Cubes_TICOI/c_x01715_y03430_interp.nc"] 

    ds_list = [xr.open_dataset(f) for f in file_list]
    merge_datasets(ds_list).to_netcdf(ds_path)

    print(f"File {ds_path} was created.")

ds_merged = xr.open_dataset(ds_path)

dx = float(ds_merged['x'][1] - ds_merged['x'][0])
dy = float(ds_merged['y'][1] - ds_merged['y'][0])
pixel_size = (dx + dy) / 2

# Target grid
x = ds_merged.x.values
y = ds_merged.y.values
X, Y = np.meshgrid(x, y)

# Flatten (for scatter)
x_1d = np.repeat(x[np.newaxis, :], y.size, axis=0).flatten()
y_1d = np.repeat(y[:, np.newaxis], x.size, axis=1).flatten()


ds_merged = ds_merged.assign_coords(dayofyear=ds_merged['mid_date'].dt.dayofyear.data, year=ds_merged['mid_date'].dt.year.data)

# Skip incomplete years
ds_merged = ds_merged.where((ds_merged['mid_date'].dt.year >= 2016) & (ds_merged['mid_date'].dt.year <= 2022), drop=True)



### ----- Background satellite image -----

tif_map = data_topo / "T32TLR_20250808T102701_TCI_60m.tif"
with rasterio.open(tif_map) as src:
    img_map = src.read([1,2,3])     # R,G,B
    bounds = src.bounds
    extent_map = [bounds.left, bounds.right, bounds.bottom, bounds.top]




### ----- DEM file -----

dem_file = data_topo / "Mt_Blanc_small_UTM32N.tif"


## Outlines glaciers
outlines_csv_path = data_topo / "mtblanc_glaciers_outlines.csv"
mtblanc_outlines = pd.read_csv(outlines_csv_path, header = 0, names = ['geometry_id', 'lon', 'lat'])

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
mtblanc_outlines['x'], mtblanc_outlines['y'] = transformer.transform(mtblanc_outlines['lon'].values, mtblanc_outlines['lat'].values)


### ----- Melt rate -----

melt_file = pd.read_csv(data_dir / "meteo" / "SafranDailyTemp_1959_2023_MtBlanc.dat", header=None, names=["temp"])
start_date = "1959-01-01"

dates = pd.date_range(start=start_date, periods=len(melt_file), freq="D")
melt_file["date"] = dates

# conversion in dataarray indexed by days:
melt_file["date"] = pd.to_datetime(melt_file["date"])
melt_da = melt_file.set_index("date")["temp"].to_xarray()


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



## ----- Discharge on Argentière -----

## Daily data (1985 - 2018)
Q_daily_Arg_bef_2018 = pd.read_excel(data_dir / "meteo" / "Q_moyen_Argentiere_1985_2018_Data_Emosson_Argentiere_Le_Tour_Gimbert.xlsx", header = None)
Q_daily_Arg_bef_2018.columns = ["date", "Q"]
Q_daily_Arg_bef_2018['date'] = pd.to_datetime(Q_daily_Arg_bef_2018['date'])

## 15-min data (2018 - 2025)
Q_15min_Arg_aft_2018 = pd.read_csv(data_dir / "meteo" / "water_discharge_2019-2021.dat", delimiter = "\s+", header = None, comment="#")
Q_15min_Arg_aft_2018.columns = ["date", "Q"]
Q_15min_Arg_aft_2018 = Q_15min_Arg_aft_2018[~Q_15min_Arg_aft_2018['date'].astype(str).str.startswith('>')]
Q_15min_Arg_aft_2018['date'] = pd.to_datetime(Q_15min_Arg_aft_2018['date'])

## Group by day
Q_15min_Arg_aft_2018['days'] = Q_15min_Arg_aft_2018['date'].dt.date
Q_daily_Arg_aft_2018 = Q_15min_Arg_aft_2018.groupby('days').mean(numeric_only=True).reset_index()
Q_daily_Arg_aft_2018.columns = ["date", "Q"]

## Filter negative data & Group the 2 timeseries
Q_daily_Arg_aft_2018 = Q_daily_Arg_aft_2018[Q_daily_Arg_aft_2018['Q']>=0]
Q_daily_mean = pd.concat([Q_daily_Arg_bef_2018, Q_daily_Arg_aft_2018])
Q_daily_mean['date'] = pd.to_datetime(Q_daily_mean['date'], errors='coerce')

# Filter between 01/2016 and 12/2022
mask = (Q_daily_mean['date'] >= '2016-01-01') & (Q_daily_mean['date'] <= '2022-12-31')
Q_filtered = Q_daily_mean[mask].copy()

# Calculates mean annual cycle (daily avg on the 7 years)
Q_filtered['doy'] = Q_filtered['date'].dt.dayofyear # Extrac doy (1 to 366)
Q_annual_cycle = Q_filtered.groupby('doy')['Q'].mean().reset_index()




## ----- Precipitation files -----

## Safran data (30 min res)
precip_safran_file = data_dir / "meteo" / "rainfall_2019-2021.dat"
df_safran_precip = pd.read_csv(precip_safran_file, sep="\s+", parse_dates=["date"], names = ["date", "precip"], header=None)

## MeteoFrance MtBlanc
precip_file = data_dir / "meteo" / "23a368d7-59e3-488d-afb7-3890f972dcd7.parquet"
df = pd.read_parquet(precip_file, columns=["NUM_POSTE", "LAT", "LON", "AAAAMMJJ", "RR"])
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
df["x_utm"], df["y_utm"] = transformer.transform(df["LON"].values, df["LAT"].values)


x_min_mf, x_max_mf = np.nanmin(x_1d), np.nanmax(x_1d)
y_min_mf, y_max_mf = np.nanmin(y_1d), np.nanmax(y_1d)

m = 1.12  # pente
b = 4715000  # intercept en mètres

# Masque des points au sud de la droite
mask_south = df["y_utm"] < (m * df["x_utm"] + b)

# Filtrer dans la zone approximative + sud de la droite
stations_mtblanc = df[
    mask_south &
    (df["y_utm"] >= y_min_mf) & (df["y_utm"] <= y_max_mf) &
    (df["x_utm"] >= x_min_mf) & (df["x_utm"] <= x_max_mf)
]["NUM_POSTE"].unique()

df_mtblanc = df[df["NUM_POSTE"].isin(stations_mtblanc)].copy()

df_mtblanc.loc[:, 'date'] = pd.to_datetime(df_mtblanc['AAAAMMJJ'], format='%Y%m%d')
df_20162022 = df_mtblanc[(df_mtblanc['date'].dt.year >= 2016) & (df_mtblanc['date'].dt.year <= 2022)].copy()

# Grouper par date et calculer la moyenne des RR
df_20162022.loc[:, 'RR'] = pd.to_numeric(df_20162022['RR'], errors='coerce')
ts_daily = df_20162022.groupby('date')['RR'].mean() 



### ----- Friction law Elmer -----

elmer_base_dir = Path("C:/Users/zellerma/Documents/PhD/Recherche/friction_long_term_alps/archive/data")

glaciers = {
    "All":  {"years": [1932,1956,1967,1982,1991,2004,2008,2012,2017,2020], "C": 0.034},
    "Arg":  {"years": [1904,1949,1952,1979,1998,2003,2008,2011,2015,2019], "C": 0.038},
    "Cor":  {"years": [1934,1983,1998,2003,2008,2017],                     "C": 0.040},
    "Gie":  {"years": [1934,1971,1985,1997,2003,2008,2013,2017,2020],      "C": 0.036},
    "GB":   {"years": [1925,1952,1967,1981],                               "C": 0.044},
    "MDG":  {"years": [1958,1979,2003,2008,2019],                          "C": 0.048},
    "StSo": {"years": [1905,1908,1952,1971,1998,2008,2019],                "C": 0.048},
    "Geb":  {"years": [1907,1953,1986,1998,2003],                          "C": 0.074},
}

step_taub_Elmer = 0.5
window_taub_Elmer = 1


### ----- Conceptual model for effective pressure -----

tau_b_mean_df=pd.read_csv(out_dir / "tau_b_mean_8_glaciers.csv")
tau_valid = tau_b_mean_df["tau_b_mean"]
slope_valid = tau_b_mean_df["slope_values"]


# Binning by slope
bin_width = 1  # 1° per bin
bins = np.arange(0, 40 + bin_width, bin_width)
bin_indices = np.digitize(slope_valid, bins)
bin_means = np.array([
    tau_valid[bin_indices == i].mean() if np.any(bin_indices == i) else np.nan
    for i in range(1, len(bins))])
bin_centers = bins[:-1] + bin_width/2
mask_bin = np.isfinite(bin_means)


slope_cut = 2.0  # degrees
slope_line = np.linspace(0, 40, 1000)
mask_line = slope_line >= slope_cut

idx = np.argsort(slope_valid) # sort by slope
x = slope_valid[idx]
y = tau_valid[idx]

from scipy.interpolate import UnivariateSpline
spl = UnivariateSpline(x, y, s=50) # Build a smoothed spline (s=smoothing factor: high s = very smooth)
tau_emp = spl(slope_line)

theta_min, theta_max = 0.4, 0.6
m = 3

CN_min = tau_emp * (1/theta_max)**(1/m)
CN_max = tau_emp * (1/theta_min)**(1/m)

# Channels control
CN_channels = 0.29 * np.tan(np.radians(slope_line))**0.47

# Intersection : indexes where CN_empirical_cst crosses CN_min & CN_max
idx_min = np.argwhere(np.diff(np.sign(CN_channels - CN_min))).flatten()
idx_max = np.argwhere(np.diff(np.sign(CN_channels - CN_max))).flatten()

slope_min_intersect = slope_line[idx_min]
slope_max_intersect = slope_line[idx_max]

slope_line = slope_line[mask_line]
tau_emp = tau_emp[mask_line]
CN_min = CN_min[mask_line]
CN_max = CN_max[mask_line]
CN_channels = CN_channels[mask_line]