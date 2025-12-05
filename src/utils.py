from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = script_dir / ".." / "data" 
geom_data_dir = script_dir / ".." / "data" / "structural"
proc_data_dir = script_dir / ".." / "data" / "processed_timeseries"
fig_dir = script_dir / ".." / "figures" 



file_list = ["../archive/data/Cubes_TICOI/c_x00980_y03920_interp.nc", "../archive/data/Cubes_TICOI/c_x01225_y03675_interp.nc", 
             "../archive/data/Cubes_TICOI/c_x01225_y03920_interp.nc", "../archive/data/Cubes_TICOI/c_x01470_y03430_1_interp.nc", 
             "../archive/data/Cubes_TICOI/c_x01470_y03430_2_interp.nc", "../archive/data/Cubes_TICOI/c_x01470_y03675_1_interp.nc", 
             "../archive/data/Cubes_TICOI/c_x01470_y03675_2_interp.nc", "../archive/data/Cubes_TICOI/c_x01715_y03430_interp.nc"] 


tif_map = data_dir / "Sentinel2/T32TLR_20250808T102701_TCI_60m.tif"
with rasterio.open(tif_map) as src:
    img_map = src.read([1,2,3])     # R,G,B
    bounds = src.bounds
    extent_map = [bounds.left, bounds.right, bounds.bottom, bounds.top]


ds_list = [xr.open_dataset(f) for f in file_list]

# Fusionner en s'assurant que le dernier fichier écrase les anciens en cas de doublons
#ds_merged = ds_list[0]
#for ds in ds_list[1:]:
#    ds_merged = ds.combine_first(ds_merged)  # Priorité au dernier dataset

#ds_merged.to_netcdf("../data/Cubes_TICOI/ds_merged.nc")
ds_merged = xr.open_dataset("../data/Cubes_TICOI/ds_merged.nc")