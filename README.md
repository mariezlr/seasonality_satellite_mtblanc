# seasonality_satellite_mtblanc
Code and processed data to reproduce the analyses and figures of the study "Slope-dependent seasonal glacier velocity cycles in the Mont Blanc massif"


WARNING: Data and code are still being organized; final version will be released upon acceptance of the manuscript.


## Raw data availability

The raw observational datasets are publicly accessible from their respective repositories:

The image used as a background for visualisation is a Sentinel2 picture from ???[date]

The DEM 


## Workflow overview

### 1. Data acquisition

Raw and processed data are stored in the data/ directory.

#### TICOI dataset


#### Additional data


### 2. Data processing and analysis

Analysis code is located in src/, structured as follows:

data_exploration.py -->   

utils.py --> 

masks.py -->

seasonal_indicators.py --> 

main.py -->

example_plots.py -->    

plots.py -->    Plot utilities that reproduce the figures of the manuscript.


### 3. Plot figures

All important paper figures can be generated using functions in plots.py

Output figures are saved in figures/