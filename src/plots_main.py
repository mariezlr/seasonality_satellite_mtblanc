from utils import *
from data_exploration import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import seaborn as sns
import matplotlib.gridspec as gridspec

velocity = np.sqrt(ds_merged['vx']**2 + ds_merged['vy']**2)
xcount = np.sqrt(ds_merged['xcount_x']**2 + ds_merged['xcount_y']**2)

# Chemin vers le fichier NetCDF
file_path = out_dir / "analysis_dataset.nc"

# Charger le Dataset
ds_analysis = xr.open_dataset(file_path)
vel_result = ds_analysis["velocity"]
vel_avg_detrended = ds_analysis["vel_detrended"]
vel_lowpass = ds_analysis["vel_lowpass"]
vel_cycle = ds_analysis["vel_cycle"]
avg_velocity = ds_analysis["avg_velocity"]
amplitude = ds_analysis["amplitude"]
amplitude_rel = ds_analysis["amplitude_rel"]
melt_cycle = ds_analysis["melt_cycle"]
slope = ds_analysis["slope"]
elevation = ds_analysis["elevation"]
max_peak_doy = ds_analysis["max_peak_doy"]
min_peak_doy = ds_analysis["min_peak_doy"]
inflex_doy = ds_analysis["inflex_doy"]
base_mask = ds_analysis["base_mask"]
result_mask = ds_analysis["mask"] & base_mask
mask_xcount = ds_analysis["mask_xcount"] & base_mask
mask_shadow = ds_analysis["mask_shadow"] & base_mask
mask_velavg = ds_analysis["mask_velavg"] & base_mask
mask_snr = ds_analysis["mask_snr"] & base_mask
mask_stable_areas = ds_analysis["mask_stable_areas"] # & base_mask

melt_summer = melt_cycle.where((melt_cycle['doy_approx'] > 166) & (melt_cycle['doy_approx'] <= 258))
avg_melt_summer = melt_summer.mean(dim=["cycle"], skipna=True)


sns.set_theme(style='whitegrid')


def plot_random_timeseries(n_samples=6, seed=42):
    """Plots timeseries for N random valid pixels"""

    random.seed(seed)
    result_mask = ds_analysis["mask"] & base_mask & (slope > 35) & (min_peak_doy > 340)

    print("Pixels valides après masque :", int(result_mask.sum()))
    # Valid pixels with non-NaN values
    valid_indices = []
    ny, nx = result_mask.sizes['y'], result_mask.sizes['x']

    for i in range(ny):
        for j in range(nx):
            if result_mask.values[i, j]:
                x0 = vel_cycle['x'].values[j]
                y0 = vel_cycle['y'].values[i]
                ts = vel_result.sel(x=x0, y=y0)
                if np.any(~np.isnan(ts.values)):
                    valid_indices.append((i, j))

    if len(valid_indices) == 0:
        print("Zero available pixel")
        return

    # Random pixels
    n = min(n_samples, len(valid_indices))
    selected = random.sample(valid_indices, n)

    cmap = plt.get_cmap('tab10', len(years))
    nrows, ncols = (np.max([1, n//3]), np.min([n, 3]))

    fig, axes = plt.subplots(nrows, ncols, sharex=True)
    if n!=1:
        axes = axes.flat


    lines_for_legend = []

    # Loop on pixels
    for k, (yi, xi) in enumerate(selected):

        if n!=1:    
            ax = axes[k]
        else:
            ax=axes

        x0 = vel_cycle['x'].values[xi]
        y0 = vel_cycle['y'].values[yi]

        ts_all_years = vel_result.sel(x=x0, y=y0)

        for j, year in enumerate(years):
            ts_year = ts_all_years.sel(mid_date=ts_all_years['mid_date.year'] == year)
            x_vals = ts_year['mid_date'].dt.dayofyear.astype(int).values
            y_vals = ts_year.values

            (line,) = ax.plot(x_vals, y_vals, color=cmap(j), alpha=0.7)

            if k == 0:  # keep lines for legend only once
                lines_for_legend.append((line, str(year)))

        # Mean seasonal vel
        ts_mean = vel_cycle.sel(x=x0, y=y0)
        ax.plot(ts_mean['cycle'], ts_mean.values, color='k', linewidth=2)

        ax.set_title(f"x={x0:.0f}, y={y0:.0f}")
        ax.grid(True)
        ax.tick_params(labelsize=12)


    fig.legend([l[0] for l in lines_for_legend],
               [l[1] for l in lines_for_legend],
               loc="center right", bbox_to_anchor=(1.02, 0.5), title="Year")

    fig.supxlabel("Day of year", fontsize=14)
    fig.supylabel("Velocity (m yr$^{-1}$)", fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    fig.savefig(fig_dir / "random_timeseries.pdf", bbox_inches='tight')
    fig.savefig(fig_dir / "random_timeseries.png", bbox_inches='tight')
    print("plot_random_timeseries Done !")
    plt.close(fig)



def plot_2pixels_ts():

    # Définir le rectangle de zoom
    xmin, xmax = 338500, 340500
    ymin, ymax = 5081500, 5083500

    # Masque rectangulaire
    mask_rect_2d = ((vel_result.x >= xmin) & (vel_result.x <= xmax)).broadcast_like(slope) & \
                    ((vel_result.y >= ymin) & (vel_result.y <= ymax)).broadcast_like(slope)

    # Sélection pixels plat et pentu
    candidate_low = mask_rect_2d & (slope < 12) & result_mask
    candidate_high = mask_rect_2d & (slope > 18) & result_mask

    candidate_low = (slope < 12) & result_mask
    candidate_high = (slope > 18) & result_mask

    y_target_low, x_target_low = 5083400, 340200
    y_target_high, x_target_high = 5082700, 339000 

    y_target_low, x_target_low = 5083400, 340200
    y_target_high, x_target_high = 5094750, 345050 

    # indices possibles selon le masque
    yi_possible_low, xi_possible_low = np.where(candidate_low.values)
    yi_possible_high, xi_possible_high = np.where(candidate_high.values)

    # distances aux coordonnées cibles
    dist_low = np.sqrt((vel_result['x'].values[xi_possible_low] - x_target_low)**2 +
                    (vel_result['y'].values[yi_possible_low] - y_target_low)**2)
    dist_high = np.sqrt((vel_result['x'].values[xi_possible_high] - x_target_high)**2 +
                        (vel_result['y'].values[yi_possible_high] - y_target_high)**2)

    # choisir le plus proche
    idx_low = dist_low.argmin()
    yi_low, xi_low = yi_possible_low[idx_low], xi_possible_low[idx_low]

    idx_high = dist_high.argmin()
    yi_high, xi_high = yi_possible_high[idx_high], xi_possible_high[idx_high]

    selected = [(yi_low, xi_low), (yi_high, xi_high)]
    labels = ["Slope < 12°", "Slope > 18°"]

    years = np.unique(vel_result['mid_date.year'].values)
    cmap = cm.get_cmap("viridis", len(years))

    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    lines_for_legend = []

    for k, (yi, xi) in enumerate(selected):
        ax = axes[k]

        x0 = vel_result['x'].values[xi]
        y0 = vel_result['y'].values[yi]

        ts_all = vel_result.sel(x=x0, y=y0)

        # Boucle sur les années
        for j, year in enumerate(years):
            ts_year = ts_all.sel(mid_date=ts_all['mid_date.year'] == year)
            x_vals = ts_year['mid_date'].dt.dayofyear.values
            y_vals = ts_year.values
            (line,) = ax.plot(x_vals, y_vals, color=cmap(j), alpha=0.7)
            if k == 0:
                lines_for_legend.append((line, str(year)))

        # Moyenne saisonnière
        ts_mean = vel_cycle.sel(x=x0, y=y0)
        ax.plot(ts_mean['cycle'], ts_mean.values, color='k', linewidth=2)

        ax.set_title(f"{labels[k]}\n(x={x0:.0f}, y={y0:.0f})")
        ax.set_xlabel("Day of year", fontsize=12)
        ax.grid(True)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel("Velocity (m yr$^{-1}$)", fontsize=12)

    fig.legend([l[0] for l in lines_for_legend],
               [l[1] for l in lines_for_legend],
               loc="center right", bbox_to_anchor=(1.02, 0.5),
               title="Year", fontsize=10)
    
    # Ajout des labels (a) et (b)
    fig.text(0.01, 0.98, '(a)', fontsize=26, fontweight='bold', va='top')
    fig.text(0.51, 0.98, '(b)', fontsize=26, fontweight='bold', va='top')


    plt.tight_layout()
    fig.savefig(fig_dir / "2pixels_ts.pdf", bbox_inches='tight')
    fig.savefig(fig_dir / "2pixels_ts.png", bbox_inches='tight')
    plt.close(fig)
    print("plot_2pixels_ts_panels Done!")


def plot_histogram_extrema_slope():

    sns.set_theme(style='whitegrid')

    # Prepare lists storing DOY values for each slope bin
    doymax_all = [[] for _ in range(n_bins)]
    doymin_all = [[] for _ in range(n_bins)]

    # Extract DOY values inside each slope interval
    for i in range(n_bins):
        mask = ((slope >= slope_bins[i]) & (slope < slope_bins[i+1]) & result_mask)

        valmax = max_peak_doy.where(mask)
        valmax = valmax.where(~np.isnan(valmax))
        doymax_all[i].extend(valmax.values.flatten())

        valmin = min_peak_doy.where(mask)
        valmin = valmin.where(~np.isnan(valmin))
        doymin_all[i].extend(valmin.values.flatten())

    # Figure
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, wspace=0.12)

    # Panel (a) max
    sub_left = subfigs[0]
    axs_left = sub_left.subplots(4, 2)

    for i in range(n_bins):
        ax = axs_left[i // 2, i % 2]
        ax.hist(doymax_all[i], bins=month_bins, color="#008080",
                edgecolor='k', alpha=0.6)
        ax.set_title(f"Slope {slope_bins[i]}–{slope_bins[i+1]}°",
                     fontsize=18)
        ax.set_xticks(mid_month_days)
        ax.set_xticklabels(month_labels_short, fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(linestyle='--', alpha=0.4)

#    axs_left[0,0].legend()
    sub_left.supxlabel("Month of maximal velocity", fontsize=24)
    sub_left.supylabel("Occurrences", fontsize=24)
    sub_left.text(0.01, 1, '(a)', fontsize=30, fontweight='bold', va='top')

    # Panel (b) min
    sub_right = subfigs[1]
    axs_right = sub_right.subplots(4, 2)

    for i in range(n_bins):
        ax = axs_right[i // 2, i % 2]
        ax.hist(doymin_all[i], bins=month_bins, color="#8b4513",
                edgecolor='k', alpha=0.6)
        ax.set_title(f"Slope {slope_bins[i]}–{slope_bins[i+1]}°",
                     fontsize=18)
        ax.set_xticks(mid_month_days)
        ax.set_xticklabels(month_labels_short, fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(linestyle='--', alpha=0.4)

#    axs_right[0,0].legend()
    sub_right.supxlabel("Month of minimal velocity", fontsize=24)
    sub_right.supylabel("Occurrences", fontsize=24)
    sub_right.text(0.01, 1, '(b)', fontsize=30, fontweight='bold', va='top')

    fig.savefig(fig_dir / "doy_distrib_max_min.pdf", bbox_inches='tight')
    fig.savefig(fig_dir / "doy_distrib_max_min.png", bbox_inches='tight')
    print("plot_histogram_extrema_slope Done !")
    plt.close(fig)


def plot_histogram_inflex_slope():

    sns.set_theme(style='whitegrid')

    # Prepare lists storing DOY values for each slope bin
    doyinflex_all = [[] for _ in range(n_bins)]


    # Extract DOY values inside each slope interval
    for i in range(n_bins):
        mask = ((slope >= slope_bins[i]) & (slope < slope_bins[i+1]) & result_mask)

        valinflex = inflex_doy.where(mask)
        valinflex = valinflex.where(~np.isnan(valinflex))
        doyinflex_all[i].extend(valinflex.values.flatten())

    # Figure
    fig, axes = plt.subplots(4, 2, figsize=(8, 10), constrained_layout=True)

    for i in range(n_bins):
        ax = axes[i // 2, i % 2]
        ax.hist(doyinflex_all[i], bins=month_bins, color='gold',
                edgecolor='k', alpha=0.8)
        ax.set_title(f"Slope {slope_bins[i]}–{slope_bins[i+1]}°",
                     fontsize=18)
        ax.set_xticks(mid_month_days)
        ax.set_xticklabels(month_labels_short, fontsize=20)
        ax.set_yticklabels(fontsize=20)

        ax.grid(linestyle='--', alpha=0.4)

#    axes[0,0].legend()

    fig.supxlabel("Month of main inflection point", fontsize=24)
    fig.supylabel("Occurrences", fontsize=24)

    fig.savefig(fig_dir / "doy_distrib_inflex.pdf", bbox_inches='tight')
    fig.savefig(fig_dir / "doy_distrib_inflex.png", bbox_inches='tight')
    print("plot_histogram_inflex_slope Done !")
    plt.close(fig)



def plot_daily_precip():
    """
    Plot daily mean velocity for low- and high-slope areas,
    together with daily precipitation.
    Pixels outside the mask are fully excluded (NaN for all dates).
    """
    # Build 2D slope masks (numpy -> xarray)
    mask_flat_2d = (slope <= 9) & result_mask
    mask_steep_2d = (slope >= 18) & result_mask
    print("Low slope pixels :", np.nansum(mask_flat_2d))
    print("High slope pixels:", np.nansum(mask_steep_2d))

    # Convert to xarray DataArray with spatial coordinates
    mask_flat = xr.DataArray(mask_flat_2d, dims=("y", "x"),
        coords={"y": vel_result.y, "x": vel_result.x})

    mask_steep = xr.DataArray(mask_steep_2d, dims=("y", "x"),
        coords={"y": vel_result.y, "x": vel_result.x})

    # Appliquer les masques sur velocity_filtered (données journalières)
    vel_low_daily = vel_result.where(mask_flat).mean(dim=["x","y"], skipna=True)
    vel_high_daily = vel_result.where(mask_steep).mean(dim=["x","y"], skipna=True)

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax1.plot(vel_low_daily['mid_date'], vel_low_daily, label=fr"Slope $\leq$ 9°", color='#3399ff')
    ax1.plot(vel_high_daily['mid_date'], vel_high_daily, label=fr"Slope $\geq$ 18°", color='#ff3300')
    ax1.set_xlabel("Date", fontsize=14)
    ax1.set_xlim(np.min(df_safran_precip['date']), np.max(df_safran_precip['date']))
    ax1.set_ylabel("Velocity (m/yr)", fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Axe y droit pour précipitations
    ax2 = ax1.twinx()
#    ax2.plot(ts_daily.index, ts_daily.values, color='turquoise', alpha=0.7, label="Daily precip from MeteoFrance")
    ax2.plot(df_safran_precip['date'], df_safran_precip['precip'], color='limegreen', alpha=0.4, label="30-min precip from Safran")
    ax2.set_ylabel("Precipitation (mm)", color='limegreen', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='limegreen')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', color='limegreen')

    plt.title("Daily velocity & precipitation (2019-2021)")

    plt.tight_layout()
    fig.savefig(fig_dir / "daily_vel_precip.png")
    fig.savefig(fig_dir / "daily_vel_precip.pdf")
    print("plot_daily_precip Done !")
    plt.close(fig)


def plot_typical_vel_melt_cycles():

    # Build 2D slope masks (numpy -> xarray)
    mask_flat_2d = (slope < 10) & result_mask
    mask_steep_2d = (slope >= 15) & result_mask

    # Convert to xarray DataArray with spatial coordinates
    mask_flat = xr.DataArray(mask_flat_2d, dims=("y", "x"),
        coords={"y": vel_result.y, "x": vel_result.x})

    mask_steep = xr.DataArray(mask_steep_2d, dims=("y", "x"),
        coords={"y": vel_result.y, "x": vel_result.x})

    # Masks applied to velocity cycles
    mean_vel_flat = vel_cycle.where(mask_flat).mean(dim=["x","y"], skipna=True)
    mean_vel_steep = vel_cycle.where(mask_steep).mean(dim=["x","y"], skipna=True)

    # Masks applied to melt rate cycles
    mean_melt_flat = melt_cycle.where(mask_flat).mean(dim=["x","y"], skipna=True)
    mean_melt_steep = melt_cycle.where(mask_steep).mean(dim=["x","y"], skipna=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(mean_vel_flat['doy_approx'], mean_vel_flat, label="slope < 10°", color='limegreen')
    ax1.plot(mean_vel_steep['doy_approx'], mean_vel_steep, label=r"slope $\geq$ 15°", color='crimson')
    ax1.set_xlabel("Day of year", fontsize=14)
    ax1.set_ylabel(r"Velocity (m yr$^{-1}$)", color='black', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Axe y droit pour débit
    ax2 = ax1.twinx()

    ax2.plot(mean_melt_flat['doy_approx'], mean_melt_flat, color='green', linestyle="--", alpha=0.7)
    ax2.plot(mean_melt_steep['doy_approx'], mean_melt_steep, color='red', linestyle = "--", alpha=0.7)
    ax2.set_ylabel(r"Melt rate (m w.e. day$^{-1}$)", color='blue', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', axis='y', linestyle='--', color='blue', alpha=0.3)

    # Ajout des labels mois en x (sur ax1)
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax1.set_xticks(month_starts)
    ax1.set_xticklabels(month_labels)

    plt.title("Mean seasonal velocity cycle by slope class & annual melt rate cycle")

    plt.tight_layout()
    fig.savefig(fig_dir / "typical_seasonal_vel_melt_cycles.pdf")
    fig.savefig(fig_dir / "typical_seasonal_vel_melt_cycles.png")
    print("plot_typical_vel_melt_cycles Done !")
    plt.close(fig)



def plot_comp_vel_melt_cycles(min_slope1, max_slope1, min_slope2, max_slope2):

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    for i in range(2):
        ax=axes[i]

        if i==0:
            color_bins = ['#00ffff', '#3399ff', '#6666ff']  
            min_slope, max_slope = min_slope1, max_slope1  
        else:
            color_bins = ['#ffcc00', '#ff3300', '#cc0000']     
            min_slope, max_slope = min_slope2, max_slope2  

        bins_temp = np.linspace(min_slope, max_slope, 4)

        # Build 2D slope masks (numpy -> xarray)
        mask_flat_2d = (slope >= bins_temp[0]) & (slope < bins_temp[1]) & result_mask
        mask_mid_2d = (slope >= bins_temp[1]) & (slope < bins_temp[2]) & result_mask
        mask_steep_2d = (slope >= bins_temp[2]) & (slope < bins_temp[3]) & result_mask

        mask_all_2d = (slope >= bins_temp[0]) & (slope < bins_temp[3]) & result_mask

        # Convert to xarray DataArray with spatial coordinates
        mask_flat = xr.DataArray(mask_flat_2d, dims=("y", "x"),
            coords={"y": vel_result.y, "x": vel_result.x})

        mask_mid = xr.DataArray(mask_mid_2d, dims=("y", "x"),
            coords={"y": vel_result.y, "x": vel_result.x})

        mask_steep = xr.DataArray(mask_steep_2d, dims=("y", "x"),
            coords={"y": vel_result.y, "x": vel_result.x})

        mask_all = xr.DataArray(mask_all_2d, dims=("y", "x"),
            coords={"y": vel_result.y, "x": vel_result.x})
        

        # Rescaling between 0 and 1 for each point (x, y)
        vel_min = vel_cycle.min(dim="cycle")
        vel_max = vel_cycle.max(dim="cycle")
        vel_rescaled = (vel_cycle - vel_min) / (vel_max - vel_min)

        # Masks applied to velocity cycles
        mean_vel_all = vel_rescaled.where(mask_all).mean(dim=["x","y"], skipna=True)[5:-5]
        mean_vel_flat = vel_rescaled.where(mask_flat).mean(dim=["x","y"], skipna=True)[5:-5]
        mean_vel_mid = vel_rescaled.where(mask_mid).mean(dim=["x","y"], skipna=True)[5:-5]
        mean_vel_steep = vel_rescaled.where(mask_steep).mean(dim=["x","y"], skipna=True)[5:-5]

        # mean_vel_all = (mean_vel_all - mean_vel_all.min()) / (mean_vel_all.max() - mean_vel_all.min())
        # mean_vel_flat = (mean_vel_flat - mean_vel_flat.min()) / (mean_vel_flat.max() - mean_vel_flat.min())
        # mean_vel_mid = (mean_vel_mid - mean_vel_mid.min()) / (mean_vel_mid.max() - mean_vel_mid.min())
        # mean_vel_steep = (mean_vel_steep - mean_vel_steep.min()) / (mean_vel_steep.max() - mean_vel_steep.min())


        # Masks applied to melt rate cycles
        mean_melt_flat = melt_cycle.where(mask_flat).mean(dim=["x","y"], skipna=True)
        mean_melt_mid = melt_cycle.where(mask_mid).mean(dim=["x","y"], skipna=True)
        mean_melt_steep = melt_cycle.where(mask_steep).mean(dim=["x","y"], skipna=True)

        #ax1.plot(mean_vel_all['doy_approx'], mean_vel_all, label=fr" {min_slope}° $\leq$ slope < {max_slope}°", color=color_vel)
        ax.plot(mean_vel_flat['doy_approx'], mean_vel_flat, color=color_bins[0], label=fr" {bins_temp[0]:.0f}° $\leq$ slope < {bins_temp[1]:.0f}°")
        ax.plot(mean_vel_mid['doy_approx'], mean_vel_mid, color=color_bins[1], label=fr" {bins_temp[1]:.0f}° $\leq$ slope < {bins_temp[2]:.0f}°")
        ax.plot(mean_vel_steep['doy_approx'], mean_vel_steep, color=color_bins[2], label=fr" {bins_temp[2]:.0f}° $\leq$ slope < {bins_temp[3]:.0f}°")
        ax.set_xlabel("Day of year", fontsize=14)
        ax.set_ylabel(r"Normalized Velocity", color='black', fontsize=14)
        ax.tick_params(axis='y', labelcolor='black')
        ax.legend(loc='upper left')
        ax.grid(True)

        # Axe y droit pour débit
        ax2 = ax.twinx()

        ax2.plot(mean_melt_flat['doy_approx'], mean_melt_flat, color=color_bins[0], linestyle="--", alpha=0.7)
        ax2.plot(mean_melt_mid['doy_approx'], mean_melt_mid, color=color_bins[1], linestyle="--", alpha=0.7)
        ax2.plot(mean_melt_steep['doy_approx'], mean_melt_steep, color=color_bins[2], linestyle = "--", alpha=0.7)
        ax2.set_ylabel(r"Melt rate (m w.e. day$^{-1}$)", color='purple', alpha=0.7, fontsize=14)
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.set_ylim(0,0.12)
    #    ax2.legend(loc='upper right')
        ax2.grid(True, which='both', axis='y', linestyle='--', color='purple', alpha=0.3)

        # Ajout des labels mois en x (sur ax1)
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)

    # Ajout des labels (a) et (b)
    fig.text(0.01, 1.01, '(a)', fontsize=26, fontweight='bold', va='top')
    fig.text(0.01, 0.51, '(b)', fontsize=26, fontweight='bold', va='top')

    #plt.title("Normalized seasonal velocity cycle by slope class & annual melt rate cycle")

    plt.tight_layout()
    fig.savefig(fig_dir / f"seasonal_velrescaled_melt_cycles{min_slope1}_{max_slope1}_{min_slope2}_{max_slope2}.pdf", bbox_inches='tight')
    fig.savefig(fig_dir / f"seasonal_velrescaled_melt_cycles{min_slope1}_{max_slope1}_{min_slope2}_{max_slope2}.png", bbox_inches='tight')
    print(f"plot_comp_vel_melt_cycles{min_slope1}_{max_slope1}_{min_slope2}_{max_slope2} Done !")
    plt.close(fig)


def plot_hist_max():

    # Prepare lists storing DOY values for each slope bin
    slope_bins = [0, 12, 18, 36]
    n_bins = len(slope_bins) - 1
    doymax_all = [[] for _ in range(n_bins)]

    # Extract DOY values inside each slope interval
    for i in range(n_bins):
        mask = ((slope >= slope_bins[i]) & (slope < slope_bins[i+1]) & result_mask)

        valmax = max_peak_doy.where(mask)
        valmax = valmax.where(~np.isnan(valmax))
        doymax_all[i].extend(valmax.values.flatten())

    fig, axes = plt.subplots(3, 1, figsize=(4, 7), constrained_layout=True)

    for i in range(n_bins):
        ax = axes[i]
        ax.hist(doymax_all[i], bins=month_bins, color="#008080",
                edgecolor='k', alpha=0.6)
        ax.set_title(f"Slope {slope_bins[i]}–{slope_bins[i+1]}°",
                     fontsize=22)
        ax.set_xticks(mid_month_days)
        ax.set_xticklabels(month_labels_short, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(linestyle='--', alpha=0.4)

#    axs_hist[0,0].legend()
    fig.supxlabel("Month of maximal velocity", fontsize=24)
    fig.supylabel("Occurrences", fontsize=24)

    fig.savefig(fig_dir / f"histogram_max_bins_6.pdf", bbox_inches='tight')
    fig.savefig(fig_dir / f"histogram_max_bins_6.png", bbox_inches='tight')
    print(f"plot_histogram_max_bins_6 Done !")
    plt.close(fig)


def plot_main_figure_phasing_amplitude(min_slope1, max_slope1, min_slope2, max_slope2, masks=True, stable_areas=False):

    fig_name = f"phasing_and_amplitude_{min_slope1}_{max_slope1}_{min_slope2}_{max_slope2}"

    if masks:
        result_mask = ds_analysis["mask"] & base_mask
    elif stable_areas:
        result_mask = base_mask & mask_stable_areas
        fig_name = f"phasing_and_amplitude_{min_slope1}_{max_slope1}_{min_slope2}_{max_slope2}_stable_areas"
    else:
        result_mask = base_mask
        fig_name = f"phasing_and_amplitude_nomask_{min_slope1}_{max_slope1}_{min_slope2}_{max_slope2}"

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)

    # Create sub-figures
    subfigs_main = fig.subfigures(1, 2, width_ratios=[0.35, 0.65], wspace=0.08)
    sub_hist = subfigs_main[0]
    subfigs = subfigs_main[1].subfigures(3, 1, height_ratios=[0.34, 0.34, 0.32], hspace=0.08)

    ### (a) Amplitude vs slope
    # Prepare lists storing DOY values for each slope bin
    slope_bins = np.arange(0, 37, 6)
    n_bins = len(slope_bins) - 1
    doymax_all = [[] for _ in range(n_bins)]
    doymin_all = [[] for _ in range(n_bins)]

    # Extract DOY values inside each slope interval
    for i in range(n_bins):
        mask = ((slope >= slope_bins[i]) & (slope < slope_bins[i+1]) & result_mask)

        valmax = max_peak_doy.where(mask)
        valmax = valmax.where(~np.isnan(valmax))
        doymax_all[i].extend(valmax.values.flatten())

        valmin = min_peak_doy.where(mask)
        valmin = valmin.where(~np.isnan(valmin))
        doymin_all[i].extend(valmin.values.flatten())

    axs_hist = sub_hist.subplots(6, 1)

    for i in range(n_bins):
        ax = axs_hist[i]
        ax.hist(doymax_all[i], bins=month_bins, color="#008080",
                edgecolor='k', alpha=0.6)
        ax.set_title(f"Slope {slope_bins[i]}–{slope_bins[i+1]}°",
                     fontsize=16)
        ax.set_xticks(mid_month_days)
        ax.set_xticklabels(month_labels_short, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(linestyle='--', alpha=0.4)

#    axs_hist[0,0].legend()
    sub_hist.supxlabel("Month of maximal velocity", fontsize=18)
    sub_hist.supylabel("Occurrences", fontsize=18)


    ### (b) & (c) seasonal velocity cycles

    ax_cycle1 = subfigs[0].subplots()
    ax_cycle2 = subfigs[1].subplots()

    for i, ax in enumerate([ax_cycle1, ax_cycle2]):

        if i==0:
            color_bins = ['#00ffff', '#3399ff', '#6666ff']  
            min_slope, max_slope = min_slope1, max_slope1  
        else:
            color_bins = ['#ffcc00', '#ff3300', '#cc0000']     
            min_slope, max_slope = min_slope2, max_slope2  

        bins_temp = np.linspace(min_slope, max_slope, 4)

        # Build 2D slope masks (numpy -> xarray)
        mask_flat_2d = (slope >= bins_temp[0]) & (slope < bins_temp[1]) & result_mask
        mask_mid_2d = (slope >= bins_temp[1]) & (slope < bins_temp[2]) & result_mask
        mask_steep_2d = (slope >= bins_temp[2]) & (slope < bins_temp[3]) & result_mask

        mask_all_2d = (slope >= bins_temp[0]) & (slope < bins_temp[3]) & result_mask

        # Convert to xarray DataArray with spatial coordinates
        mask_flat = xr.DataArray(mask_flat_2d, dims=("y", "x"),
            coords={"y": vel_result.y, "x": vel_result.x})

        mask_mid = xr.DataArray(mask_mid_2d, dims=("y", "x"),
            coords={"y": vel_result.y, "x": vel_result.x})

        mask_steep = xr.DataArray(mask_steep_2d, dims=("y", "x"),
            coords={"y": vel_result.y, "x": vel_result.x})

        mask_all = xr.DataArray(mask_all_2d, dims=("y", "x"),
            coords={"y": vel_result.y, "x": vel_result.x})
        

        # Rescaling between 0 and 1 for each point (x, y)
        vel_min = vel_cycle.min(dim="cycle")
        vel_max = vel_cycle.max(dim="cycle")
        vel_rescaled = (vel_cycle - vel_min) / (vel_max - vel_min)

        # Masks applied to velocity cycles
        mean_vel_all = vel_rescaled.where(mask_all).mean(dim=["x","y"], skipna=True)[5:-5]
        mean_vel_flat = vel_rescaled.where(mask_flat).mean(dim=["x","y"], skipna=True)[5:-5]
        mean_vel_mid = vel_rescaled.where(mask_mid).mean(dim=["x","y"], skipna=True)[5:-5]
        mean_vel_steep = vel_rescaled.where(mask_steep).mean(dim=["x","y"], skipna=True)[5:-5]

        # mean_vel_all = (mean_vel_all - mean_vel_all.min()) / (mean_vel_all.max() - mean_vel_all.min())
        # mean_vel_flat = (mean_vel_flat - mean_vel_flat.min()) / (mean_vel_flat.max() - mean_vel_flat.min())
        # mean_vel_mid = (mean_vel_mid - mean_vel_mid.min()) / (mean_vel_mid.max() - mean_vel_mid.min())
        # mean_vel_steep = (mean_vel_steep - mean_vel_steep.min()) / (mean_vel_steep.max() - mean_vel_steep.min())

        # Masks applied to melt rate cycles
        mean_melt_flat = melt_cycle.where(mask_flat).mean(dim=["x","y"], skipna=True)
        mean_melt_mid = melt_cycle.where(mask_mid).mean(dim=["x","y"], skipna=True)
        mean_melt_steep = melt_cycle.where(mask_steep).mean(dim=["x","y"], skipna=True)

        #ax1.plot(mean_vel_all['doy_approx'], mean_vel_all, label=fr" {min_slope}° $\leq$ slope < {max_slope}°", color=color_vel)
        ax.plot(mean_vel_flat['doy_approx'], mean_vel_flat, color=color_bins[0], label=fr" {bins_temp[0]:.0f}° $\leq$ slope < {bins_temp[1]:.0f}°")
        ax.plot(mean_vel_mid['doy_approx'], mean_vel_mid, color=color_bins[1], label=fr" {bins_temp[1]:.0f}° $\leq$ slope < {bins_temp[2]:.0f}°")
        ax.plot(mean_vel_steep['doy_approx'], mean_vel_steep, color=color_bins[2], label=fr" {bins_temp[2]:.0f}° $\leq$ slope < {bins_temp[3]:.0f}°")
        ax.set_xlabel("Day of year", fontsize=18)
        ax.set_ylabel(r"Normalized Velocity", color='black', fontsize=18)
        ax.tick_params(axis='y', labelcolor='black')
        ax.legend(loc='upper left')
        ax.grid(True)

        # Axe y droit pour débit
        ax2 = ax.twinx()

        ax2.plot(mean_melt_flat['doy_approx'], mean_melt_flat, color=color_bins[0], linestyle="--", alpha=0.7)
        ax2.plot(mean_melt_mid['doy_approx'], mean_melt_mid, color=color_bins[1], linestyle="--", alpha=0.7)
        ax2.plot(mean_melt_steep['doy_approx'], mean_melt_steep, color=color_bins[2], linestyle = "--", alpha=0.7)
        ax2.set_ylabel(r"Melt rate (m w.e. day$^{-1}$)", color='purple', fontsize=18)
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.set_ylim(0,0.12)
    #    ax2.legend(loc='upper right')
        ax2.grid(True, which='both', axis='y', linestyle='--', color='purple', alpha=0.3)

        # Ajout des labels mois en x (sur ax1)
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)


    ### (d) Amplitude vs slope
    slope_vals = slope.where(result_mask).values.flatten()
    relampl_vals = amplitude_rel.where(result_mask).values.flatten()
    
    # between mid-June and mid-September
    melt_vals = avg_melt_summer.where(result_mask).values.flatten()

    # Filter slopes <36°
    mask = (slope_vals >= 0) & (slope_vals <= 36)
    slope_vals_filtered = slope_vals[mask]
    relampl_vals_filtered = relampl_vals[mask]
    melt_vals_filtered = melt_vals[mask]

    # Moyenne glissante
    step = 1      # pas entre les centres (1°)
    window = 3    # largeur de la fenêtre glissante (3°)

    centers, mean_relampl = moving_average(relampl_vals_filtered, slope_vals_filtered, step, window)
    centers, mean_melt = moving_average(melt_vals_filtered, slope_vals_filtered, step, window)

    # Axe gauche : vitesse et amplitude
    ax_amp = subfigs[2].subplots()

    ax_amp.plot(centers, mean_relampl, color='green', label='Relative amplitude')
    ax_amp.set_xlabel('Slope (°)', fontsize=18)
    ax_amp.set_ylabel('Relative Amplitude', color="green", fontsize=18)
    ax_amp.tick_params(axis='y', labelcolor='green')
    ax_amp.legend(loc='upper left')
    ax_amp.grid(True, which='both', linestyle='--', color='green', alpha=0.3)

    ax4 = ax_amp.twinx()
    ax4.plot(centers, mean_melt, color='purple', linestyle="--", label='Summer melt')
    ax4.set_ylabel(r"Mean summer melt rate" "\n" r"(m w.e. day$^{-1}$)", color='purple', fontsize=18)
    ax4.tick_params(axis='y', labelcolor='purple')
    ax4.legend(loc='upper right')
    ax4.grid(True, which='both', axis='y', linestyle='--', color='purple', alpha=0.3)


    # Ajout des labels (a) et (b)
    fig.text(0.01, 1.02, '(a)', fontsize=26, fontweight='bold', va='top')
    fig.text(0.35, 1.02, '(b)', fontsize=26, fontweight='bold', va='top')
    fig.text(0.35, 0.68, '(c)', fontsize=26, fontweight='bold', va='top')
    fig.text(0.35, 0.34, '(d)', fontsize=26, fontweight='bold', va='top')

    fig.savefig(fig_dir / f"{fig_name}.pdf", bbox_inches='tight')
    fig.savefig(fig_dir / f"{fig_name}.png", bbox_inches='tight')
    print(f"plot_main_figure_phasing_amplitude_{min_slope1}_{max_slope1}_{min_slope2}_{max_slope2} Done !")
    plt.close(fig)



from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

def plot_map_serac_fall_slope_bins():

    fig = plt.figure(figsize=(11, 8))

    # Create sub-grids
    gs_main = gridspec.GridSpec(1, 2, width_ratios=[2, 3], wspace=0.25)

    gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0], height_ratios=[0.7, 0.3], hspace=0.15)
    gs_right = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[1])

    # Create axes for 3 figures
    ax1 = fig.add_subplot(gs_left[0])
    ax2 = fig.add_subplot(gs_left[1])
    ax3 = fig.add_subplot(gs_right[0])


    ### AX1 : Map zoom Serac fall MDG ###
    # Extraire les coordonnées et les valeurs
    doy_flat = max_peak_doy.values.flatten()

    # Appliquer les masques sur les données aplaties
    winter_spring_mask = ((doy_flat >= 355) | (doy_flat <= 170))
    summer_autumn_mask = ((doy_flat >= 171) & (doy_flat <= 354))

    # Créer un tableau de couleurs en fonction des masques
    colors = np.where(winter_spring_mask, '#3399ff', '#ff3300')

    # Tracer avec scatter
    ax1.scatter(x_1d, y_1d, c=colors, s=3, rasterized=True)

    ax1.set_xlim(338000, 341000)     # ax.set_xlim(335000, 342000)
    ax1.set_ylim(5081000, 5084000)   # ax.set_ylim(5078000, 5085000)

    elev_zoom = elevation.where(
        (elevation.x >= 338000) &
        (elevation.x <= 341000) &
        (elevation.y >= 5081000) &
        (elevation.y <= 5084000), drop=True)

    levels = np.arange(2000, 5000, 50)
    contour = ax1.contour(elev_zoom.x, elev_zoom.y, elev_zoom, levels=levels, colors='black', alpha=0.9, linewidths=0.8)
    
    label_levels = contour.levels[::4]
    labels = ax1.clabel(contour, levels=label_levels, fmt='%d m', fontsize=9)

    for txt in labels:
        txt.set_path_effects([
            pe.Stroke(linewidth=2, foreground='white'),
            pe.Normal()
        ])

    ax1.scatter([], [], c='#3399ff', label='Max in winter / spring')
    ax1.scatter([], [], c='#ff3300', label='Max in summer / autumn')
    ax1.legend(loc='upper left')

    ax1.set_title("")

    ax1.plot([point[0] for point in zoom_points], [point[1] for point in zoom_points], linewidth = 4, color = "purple")
    base_mask.plot.contourf(ax=ax1, levels=[0.5, 1.5], colors=['black', 'none'], add_colorbar=False)


    ax1.tick_params(axis='x', labelrotation=40)

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect("equal")
    ax1.grid(False)
    

    ### AX2 : Longitudinal cross-section ### 

    # Interpoler les masques sur la flowline
    winter_spring_flowline = griddata((x_1d, y_1d), winter_spring_mask, 
                                      (x_flowline, y_flowline), method='nearest')


    colors_flowline = np.where(winter_spring_flowline, '#3399ff', '#ff3300')

    ax1.plot(x_flowline, y_flowline, color='purple', linewidth=2)

    # Position au milieu
    mid_index = len(x_flowline) // 5
    x_arrow = x_flowline[mid_index]
    y_arrow = y_flowline[mid_index]

    # Direction locale
    dx = x_flowline[mid_index + 1] - x_flowline[mid_index - 1]
    dy = y_flowline[mid_index + 1] - y_flowline[mid_index - 1]

    # Normalisation (important pour que la taille soit constante)
    norm = np.hypot(dx, dy)
    dx /= norm
    dy /= norm

    # Longueur physique de la flèche (en mètres ici)
    L = 150

    arrow = FancyArrowPatch(
        (x_arrow - L*dx, y_arrow - L*dy),
        (x_arrow + L*dx, y_arrow + L*dy),
        arrowstyle='-|>',   # plus propre que '->'
        mutation_scale=20,  # taille de la tête
        color='purple',
        linewidth=3
    )

    ax1.add_patch(arrow)


    z_flowline = griddata((x_1d, y_1d), elevation.values.flatten(), 
                      (x_flowline, y_flowline), method='linear')
    
    # Tracer la flowline colorée sur ax2
    ax2.scatter(distances, z_flowline, c=colors_flowline, s=10, edgecolor='none')
    ax2.plot(distances, z_flowline, color='k', alpha=0.5, label='Flowline')

    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Elevation (m)")

    ax2.grid(linestyle='--')


    ### AX3 : Map slope bins  masks ###
    # Définir les classes de pente (en degrés)
    slope_bins = np.arange(0, 65, 10)
    n_bins = len(slope_bins) - 1
    color_bins = ['#00ffff', '#3399ff', '#6666ff', '#66cc66', '#ffcc00', '#ff3300', '#cc0000']

    # Fond satellite
    ax3.imshow(np.moveaxis(img_map, 0, -1), extent=extent_map, origin='upper')

    import matplotlib.patches as mpatches
    legend_patches = []

    # Boucle sur chaque classe de pente
    for i in range(n_bins):
        
        mask = ((slope >= slope_bins[i]) & (slope < slope_bins[i+1]) & result_mask)
        mask.plot.contourf(ax=ax3, levels=[0.5, 1.5], colors=['none', color_bins[i]], add_colorbar=False)
    
        legend_patches.append(mpatches.Patch(color=color_bins[i], label=f"{slope_bins[i]}° $\leq$ slope < {slope_bins[i+1]}°"))

    ax3.plot([point[0] for point in zoom_points], [point[1] for point in zoom_points], linewidth = 2, color = "purple")
    ax3.plot(x_flowline, y_flowline, color='purple', linewidth=1.5)

    ax3.set_xlim(np.nanmin(x_1d) + 9000, np.nanmax(x_1d) - 9000)
    ax3.set_ylim(np.nanmin(y_1d), np.nanmax(y_1d))

    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")

    ax3.legend(handles=legend_patches, markerscale=2, fontsize = 8, loc='upper left')
    ax3.grid(False)


    # Ajout des labels (a) et (b)
    fig.text(0.003, 0.9, '(a)', fontsize=22, fontweight='bold', va='top')
    fig.text(0.003, 0.36, '(b)', fontsize=22, fontweight='bold', va='top')
    fig.text(0.39, 0.9, '(c)', fontsize=22, fontweight='bold', va='top')


    fig.savefig(fig_dir / "map_serac_fall_slope_bins.pdf", bbox_inches='tight')
    fig.savefig(fig_dir / "map_serac_fall_slope_bins.png", bbox_inches='tight')
    print("plot_map_serac_fall_slope_bins Done !")
    plt.close(fig)



def plot_map_slope_bins(bins = True, zoom = False):

    fig, ax = plt.subplots()

    # Définir les classes de pente (en degrés)
    slope_bins = np.arange(0, 65, 10)
    n_bins = len(slope_bins) - 1
    color_bins = ['#00ffff', '#3399ff', '#6666ff', '#66cc66', '#ffcc00', '#ff3300', '#cc0000']

    # Fond satellite
    ax.imshow(np.moveaxis(img_map, 0, -1), extent=extent_map, origin='upper')

    import matplotlib.patches as mpatches
    legend_patches = []

    # Boucle sur chaque classe de pente
    if bins:
        for i in range(n_bins):
            
            mask = ((slope >= slope_bins[i]) & (slope < slope_bins[i+1]) & result_mask)
            mask.plot.contourf(ax=ax, levels=[0.5, 1.5], colors=['none', color_bins[i]], add_colorbar=False)
        
            legend_patches.append(mpatches.Patch(color=color_bins[i], label=f"{slope_bins[i]}° $\leq$ slope < {slope_bins[i+1]}°"))

    if zoom:    
        ax.plot([point[0] for point in zoom_points], [point[1] for point in zoom_points], linewidth = 2, color = "purple")
        ax.plot(x_flowline, y_flowline, color='purple', linewidth=1.5)

    ax.set_xlim(np.nanmin(x_1d) + 9000, np.nanmax(x_1d) - 9000)
    ax.set_ylim(np.nanmin(y_1d), np.nanmax(y_1d))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.legend(handles=legend_patches, markerscale=2, fontsize = 8, loc='upper left')
    ax.grid(False)


    fig.savefig(fig_dir / "map_slope_bins.pdf", bbox_inches='tight')
    fig.savefig(fig_dir / "map_slope_bins.png", bbox_inches='tight')
    print("plot_map_slope_bins Done !")
    plt.close(fig)




def plot_altitude_analysis():

    z_vals = elevation.where(result_mask).values.flatten()
    max_vals = max_peak_doy.where(result_mask).values.flatten()
    min_vals = min_peak_doy.where(result_mask).values.flatten()
    avg_vals = avg_velocity.where(result_mask).values.flatten()
    ampl_vals = amplitude.where(result_mask).values.flatten()
    relampl_vals = amplitude_rel.where(result_mask).values.flatten()

    # Moyenne glissante
    step = 10       # pas entre les centres (10 m)
    window = 200    # largeur de la fenêtre glissante (200 m)

    centers = np.arange(np.nanmin(z_vals), np.nanmax(z_vals)+step, step)

    centers, mean_maxdoy = moving_average(max_vals, z_vals, step, window)
    centers, mean_mindoy = moving_average(min_vals, z_vals, step, window)
    centers, mean_avg = moving_average(avg_vals, z_vals, step, window)
    centers, mean_ampl = moving_average(ampl_vals, z_vals, step, window)
    centers, mean_relampl = moving_average(relampl_vals, z_vals, step, window)
    

    fig, ax1 = plt.subplots(figsize=(6,5))

    # Axe gauche : vitesse et amplitude
    ax1.plot(centers, mean_avg, color='limegreen', label='Average velocity')
    ax1.plot(centers, mean_ampl, color='darkorange', label='Amplitude')
    ax1.plot(centers, mean_relampl, color='crimson', label='Relative amplitude')
    ax1.set_xlabel('Elevation (m)')
    ax1.set_ylabel('Velocity / Amplitude (m/yr)')
    ax1.grid(True, which='both', linestyle='--')

    # Axe droit : DOY transformé en mois
    ax2 = ax1.twinx()
    ax2.plot(centers, mean_maxdoy, color='skyblue', label='DOY of max velocity')
    ax2.plot(centers, mean_mindoy, color='violet', label='DOY of min velocity')
    ax2.set_ylabel('Month of min/max velocity')
    ax2.grid(False)

    # Convertir DOY en mois pour ticks
    month_ticks = np.arange(1,13)
    doy_ticks = month_ticks * 30.4
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    ax2.set_yticks(doy_ticks)
    ax2.set_yticklabels(month_labels)

    # Légendes
    ax1.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    fig.savefig(fig_dir / "cycles_by_altitude.pdf")
    fig.savefig(fig_dir / "cycles_by_altitude.png")
    print("plot_altitude_analysis Done !")
    plt.close(fig)



def plot_relative_amplitude_vs_slope():

    slope_vals = slope.where(result_mask).values.flatten()
    relampl_vals = amplitude_rel.where(result_mask).values.flatten()

    # between mid-June and mid-September
    avg_melt_summer = melt_summer.mean(dim=["cycle"], skipna=True)
    melt_vals = avg_melt_summer.where(result_mask).values.flatten()


    # Filter slopes <40°
    mask = (slope_vals >= 0) & (slope_vals <= 40)
    slope_vals_filtered = slope_vals[mask]
    relampl_vals_filtered = relampl_vals[mask]
    melt_vals_filtered = melt_vals[mask]

    # Moyenne glissante
    step = 1       # pas entre les centres (1°)
    window = 8    # largeur de la fenêtre glissante (5°)

    centers, mean_relampl = moving_average(relampl_vals_filtered, slope_vals_filtered, step, window)
    centers, mean_melt = moving_average(melt_vals_filtered, slope_vals_filtered, step, window)


    fig, ax = plt.subplots(figsize=(6,4))

    # Axe gauche : vitesse et amplitude
    ax.plot(centers, mean_relampl, color='green', label='Relative amplitude')
    ax.set_xlabel('Slope (°)')
    ax.set_ylabel('Relative Amplitude', color="green")
    ax.tick_params(axis='y', labelcolor='green')
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', color='green', alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(centers, mean_melt, color='purple', linestyle="--", label='Summer melt')
    ax2.set_ylabel(r"Mean summer melt rate (m w.e. day$^{-1}$)", color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', axis='y', linestyle='--', color='purple', alpha=0.3)


    plt.tight_layout()
    fig.savefig(fig_dir / "amplitude_vs_slope.pdf")
    fig.savefig(fig_dir / "amplitude_vs_slope.png")
    print("plot_relative_amplitude_vs_slope Done !")
    plt.close(fig)



def plot_conceptual_effective_pressure_model():

    print(slope_min_intersect, slope_max_intersect)

    fig, ax = plt.subplots(figsize=(7,5))

    # ax.plot(slope_line, tau_emp, linestyle='--', color='green', label=r'Average $\tau_b$')#, label=r'Average $\tau_b$ from Elmer/Ice simulations')

    ax.plot(slope_line, CN_min, linestyle='-', color='orange')
    ax.plot(slope_line, CN_max, linestyle='-', color='orange')
    ax.fill_between(slope_line, CN_min, CN_max,
        color="orange", alpha=0.3, label=r"$CN_{cavities}$ = $\tau_b(1/\theta)^{1/3}$ (in summer); $\theta \in $[0.4;0.6]")

    ax.plot(slope_line, CN_channels_min, linestyle='-', color='blue')
    ax.plot(slope_line, CN_channels_max, linestyle='-', color='blue')
    ax.fill_between(slope_line, CN_channels_min, CN_channels_max,
        color="blue", alpha=0.3, label=r'$CN_{conduits}$ = f tan$(\alpha)^{0.47}$ (in winter); f$ \in $[0.28;0.30]')
    

    # Calculer la différence entre les courbes orange (CN_max) et bleue (CN_channels_max)
    diff = (CN_channels_max + CN_channels_min)/2 - (CN_max + CN_min)/2
    diff_normalized = (diff - np.abs(diff).min()) / (np.abs(diff).max() - np.abs(diff).min())  # Normaliser entre 0 et 1


    # Créer une matrice 2D pour le dégradé horizontal
    y_min = min(CN_min.min(), CN_channels_min.min())
    y_max = max(CN_max.max(), CN_channels_max.max())
    Y = np.linspace(y_min, y_max, 100)  # 100 points verticaux couvrant toute la hauteur
    X, _ = np.meshgrid(slope_line, Y)  # Grille 2D
    Z = np.tile(diff_normalized, (len(Y), 1))  # Répéter la différence normalisée pour chaque ligne verticale
    
    print(np.min(Z), np.max(Z))

    # Tracer le dégradé en arrière-plan
    Z = np.abs(Z)
    ax.pcolormesh(X, Y, Z, cmap="Greys_r", vmin=-0.4, vmax=0.4, alpha=0.5, shading='auto', zorder=0)
    print()

    ax.set_xlabel("Surface slope (°)")
    ax.set_ylabel(fr"CN (MPa)")

    # ax.set_xscale("log")
    # ax.set_yscale("log")

    # x_ticks = np.arange(5, 45, 5)
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels([f"{x:.0f}" for x in x_ticks])

    # y_ticks = [0.05, 0.06, 0.10, 0.15, 0.20, 0.25]
    # ax.set_yticks(y_ticks)
    # ax.set_yticklabels([f"{y:.2f}" for y in y_ticks])

    ax.legend(loc = "upper left")
    ax.grid(linestyle="--")

    plt.tight_layout()
    fig.savefig(fig_dir / "N_vs_slope.pdf")
    fig.savefig(fig_dir / "N_vs_slope.png")
    print("plot_conceptual_effective_pressure_model Done !")
    plt.close(fig)


import matplotlib.ticker as mticker
from matplotlib.ticker import FixedLocator, FuncFormatter

def plot_friction_laws():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))

    print(slope_min_intersect, slope_max_intersect)
    # ax.plot(slope_line, tau_emp, linestyle='--', color='green', label=r'Average $\tau_b$')#, label=r'Average $\tau_b$ from Elmer/Ice simulations')

    ax1.plot(slope_line, CN_min, linestyle='-', color='orange')
    ax1.plot(slope_line, CN_max, linestyle='-', color='orange')
    ax1.fill_between(slope_line, CN_min, CN_max,
        color="orange", alpha=0.3, label=r"$CN_{cavities}$ = $\tau_b(1/\theta)^{1/3}$ (in summer); $\theta \in $[0.4;0.6]")

    ax1.plot(slope_line, CN_channels_min, linestyle='-', color='blue')
    ax1.plot(slope_line, CN_channels_max, linestyle='-', color='blue')
    ax1.fill_between(slope_line, CN_channels_min, CN_channels_max,
        color="blue", alpha=0.3, label=fr'$CN_{{conduits}}$ = f tan$(\alpha)^{{0.47}}$ (in winter); f$ \in $[{f_min:.2f};{f_max:.2f}]')
    

    # Calculer la différence entre les courbes orange (CN_max) et bleue (CN_channels_max)
    diff = (CN_channels_max + CN_channels_min)/2 - (CN_max + CN_min)/2
    diff_normalized = (diff - np.abs(diff).min()) / (np.abs(diff).max() - np.abs(diff).min())  # Normaliser entre 0 et 1


    # Créer une matrice 2D pour le dégradé horizontal
    y_min = min(CN_min.min(), CN_channels_min.min())
    y_max = max(CN_max.max(), CN_channels_max.max())
    Y = np.linspace(y_min, y_max, 100)  # 100 points verticaux couvrant toute la hauteur
    X, _ = np.meshgrid(slope_line, Y)  # Grille 2D
    Z = np.tile(diff_normalized, (len(Y), 1))  # Répéter la différence normalisée pour chaque ligne verticale
    
    print(np.min(Z), np.max(Z))

    # Tracer le dégradé en arrière-plan
    # ax.pcolormesh(X, Y, Z, cmap="RdYlBu", vmin=-1, vmax=1, alpha=0.5, shading='auto', zorder=0)
    Z = np.abs(Z)
    ax1.pcolormesh(X, Y, Z, cmap="Greys_r", vmin=-0.4, vmax=0.4, alpha=0.5, shading='auto', zorder=0)


    ax1.set_xlabel("Surface slope (°)")
    ax1.set_ylabel(fr"CN (MPa)")

    ax1.legend(loc = "upper left")
    ax1.grid(linestyle="--")



    sliding_vel = np.arange(1, 500, 0.1)
    As_winter = 7500
    As_summer = As_winter /(1-0.5)

    CN_6 = 0.29 * np.tan(np.radians(6))**0.47
    # CN_12 = 0.29 * np.tan(np.radians(12))**0.47
    CN_24 = 0.29 * np.tan(np.radians(24))**0.47

    ax2.hlines(CN_6, np.min(sliding_vel), np.max(sliding_vel), color = "k", linewidth=0.5, linestyle="--")
    ax2.text(1.1, 0.98 * CN_6, r"$CN(6^\circ)$", va="top", ha="left", fontsize=9)
    ax2.hlines(CN_24, np.min(sliding_vel), np.max(sliding_vel), color = "k", linewidth=0.5, linestyle="--")
    ax2.text(1.1, 0.98 * CN_24, r"$CN(24^\circ)$", va="top", ha="left", fontsize=9)

    taub_summer = [power_law(u, As_summer) for u in sliding_vel]
    taub_winter_6 = [cavitation_law(u, CN_6, 1, As_winter) for u in sliding_vel]
    # taub_winter_12 = [cavitation_law(u, CN_12, 1, As_winter) for u in sliding_vel]
    taub_winter_24 = [cavitation_law(u, CN_24, 1, As_winter) for u in sliding_vel]

    ax2.plot(sliding_vel, taub_summer, color="#E6A700", linewidth=5, alpha = 0.9, label="Summer")
    ax2.plot(sliding_vel, taub_winter_6, color="#2F7FEA", linewidth=5, alpha = 0.9)
    # ax2.plot(sliding_vel, taub_winter_12, color="#A4BDF4", linewidth=5, alpha = 0.9)
    ax2.plot(sliding_vel, taub_winter_24, color="#4FA3F7", linewidth=5, alpha = 0.9, label="Winter")

    u_arrow = 100
    tau_6 = cavitation_law(u_arrow, CN_6, 1, As_winter)
    ax2.annotate("Slope = 6°", xy=(u_arrow, tau_6), xytext=(u_arrow*0.65, tau_6*1.2), color="#2F7FEA", fontweight='bold',
        arrowprops=dict(arrowstyle="->", linewidth=1, color="#2F7FEA"), fontsize=16)


    # u_arrow = 5
    # tau_12 = cavitation_law(u_arrow, CN_12, 1, As_winter)
    # ax2.annotate("Slope = 12°", xy=(u_arrow, tau_12), xytext=(u_arrow*0.4, tau_12*1.12), color="#A4BDF4", fontweight='bold',
    #     arrowprops=dict(arrowstyle="->", linewidth=1, color="#A4BDF4"), fontsize=16)


    u_arrow = 50
    tau_24 = cavitation_law(u_arrow, CN_24, 1, As_winter)
    ax2.annotate("Slope = 24°", xy=(u_arrow, tau_24), xytext=(u_arrow*0.2, tau_24*1.15), color="#4FA3F7", fontweight='bold',
        arrowprops=dict(arrowstyle="->", linewidth=1, color="#4FA3F7"), fontsize=16)

    basal_shear_stress = np.arange(0.04, 0.25, 10**(-8))

    inv_summer = interp1d(np.array(taub_summer), sliding_vel, bounds_error=False, fill_value=np.nan)
    inv_winter_6 = interp1d(np.array(taub_winter_6), sliding_vel, bounds_error=False, fill_value=np.nan)
    inv_winter_24 = interp1d(np.array(taub_winter_24), sliding_vel, bounds_error=False, fill_value=np.nan)

    ub_mean_6 = (1/2) * inv_summer(basal_shear_stress) + (1/2) * inv_winter_6(basal_shear_stress)
    ub_mean_24 = (1/2) * inv_summer(basal_shear_stress) + (1/2) * inv_winter_24(basal_shear_stress)
    
    ax2.plot(ub_mean_6, basal_shear_stress, color="orangered", linestyle="--", label="Annual mean")
    ax2.plot(ub_mean_24, basal_shear_stress, color="orangered", linestyle="--")

    tau_arrow = 0.09
    u_summer = inv_summer(tau_arrow)
    u_winter = inv_winter_6(tau_arrow)
    ax2.annotate("", xy=(u_summer, tau_arrow), xytext=(u_winter, tau_arrow),
        arrowprops=dict(arrowstyle="->", linewidth=1, color="black"), fontsize=12)

    ax2.text(u_winter*0.9, tau_arrow*0.88, "Winter faster", fontweight='bold', rotation=15, ha="center", va="bottom", fontsize=12, color="black")

    tau_arrow = 0.11
    u_summer = inv_summer(tau_arrow)
    u_winter = inv_winter_24(tau_arrow)
    ax2.annotate("", xy=(u_summer, tau_arrow), xytext=(u_winter, tau_arrow),
        arrowprops=dict(arrowstyle="->", linewidth=1, color="black"), fontsize=12)

    ax2.text(u_winter*0.9, tau_arrow*0.94, "Winter slower", fontweight='bold', rotation=50, ha="center", va="bottom", fontsize=12, color="black")

    ax2.legend(loc="center left", bbox_to_anchor=(0, 0.75))

    ax2.set_xlim(1, 200)
    ax2.set_ylim(0.05, 0.24)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.margins(0)


    ax2.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs='auto'))
    ax2.grid(which='both', color='0.9', linewidth=0.6)
    ax2.tick_params(which='major', direction='in', top=True, bottom=True, left=True, right=True, length=5, width=0.8)
    ax2.tick_params(which='minor', direction='in', top=True, bottom=True, left=True, right=True, length=3, width=0.8)

    y_ticks = [0.05, 0.1, 0.15, 0.2]
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels([f"{y:.2f}" for y in y_ticks])
    ax2.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.get_yaxis().set_minor_formatter(plt.NullFormatter())
    ax2.set_xlabel(r'Basal sliding velocity $(m \cdot yr^{-1})$')
    ax2.set_ylabel(r'Basal shear stress (MPa)')

   # Ajout des labels (a) et (b)
    fig.text(0.01, 1.01, '(a)', fontsize=22, fontweight='bold', va='top')
    fig.text(0.51, 1.01, '(b)', fontsize=22, fontweight='bold', va='top')


    plt.tight_layout()
    fig.savefig(fig_dir / "CN_conceptual_model.pdf", bbox_inches='tight')
    fig.savefig(fig_dir / "CN_conceptual_model.png", bbox_inches='tight')
    print("plot_friction_laws Done !")
    plt.close(fig) 


def plot_CN_enveloppe():

    # ligne de pente pour tracer l'enveloppe
    slope_line = np.linspace(min(slope_vals_Elmer)*0.8, max(slope_vals_Elmer)*1.2, 200)

    # figure
    fig, ax = plt.subplots(figsize=(6,5))

    # points
    ax.scatter(slope_vals_Elmer, CN_vals_Elmer, color="black", zorder=3)

    CN_channels_min = f_min * np.tan(np.radians(slope_line))**0.47
    CN_channels_max = f_max * np.tan(np.radians(slope_line))**0.47

    # enveloppe
    ax.plot(slope_line, CN_channels_min, color="blue")
    ax.plot(slope_line, CN_channels_max, color="blue")

    ax.fill_between(
        slope_line,
        CN_channels_min,
        CN_channels_max,
        color="blue",
        alpha=0.3,
        label=fr'$CN = f \tan(\alpha)^{{0.47}},\ f \in [{f_min:.2f},{f_max:.2f}]$'
    )

    # axes
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Mean slope (°)")
    ax.set_ylabel("CN (MPa)")

    ax.grid(True, which="both", linestyle="dotted")
    ax.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / "CN_enveloppe.pdf", bbox_inches='tight')
    fig.savefig(fig_dir / "CN_enveloppe.png", bbox_inches='tight')
    print("plot_CN_enveloppe Done !")
    plt.close(fig) 

def plot_masks():

    # Créer une figure 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Liste des masques et leurs titres
    masks = [mask_xcount, mask_shadow, mask_velavg, mask_snr]
    titles = ["Mask Xcount", "Mask Shadow", "Mask Average Velocity", "Mask SNR"]

    # Tracer chaque masque
    for i, ax in enumerate(axes.flat):
        # Afficher le fond satellite
        ax.imshow(np.moveaxis(img_map, 0, -1), extent=extent_map, origin='upper')

        # Afficher le masque (noir = valide, transparent = invalide)
        current_mask = masks[i]
        current_mask.plot.contourf(ax=ax, levels=[0.5, 1.5], colors=['none', 'black'], add_colorbar=False)
        
        # Ajouter un titre
        ax.set_title(titles[i], fontsize=22)
        ax.axis('off')

        ax.set_xlim(np.nanmin(x_1d), np.nanmax(x_1d))
        ax.set_ylim(np.nanmin(y_1d), np.nanmax(y_1d))

    plt.tight_layout()
    fig.savefig(fig_dir / "masks_map.pdf", bbox_inches='tight')
    fig.savefig(fig_dir / "masks_map.png", bbox_inches='tight')
    print("plot_masks Done !")
    plt.close(fig)


def print_stats():
    mask_valid = result_mask 
#    mask_valid = mask_xcount & mask_shadow & mask_snr

    avg_velocity_masked = avg_velocity.where(mask_valid)
    amplitude_masked = amplitude.where(mask_valid)
    slope_masked = slope.where(mask_valid)
    melt_cycle_masked = melt_cycle.where(mask_valid)

    # Calculer les statistiques pour les vitesses moyennes annuelles
    mean_velocity = avg_velocity_masked.mean().values
    median_velocity = np.nanmedian(avg_velocity_masked.values)
    q1_velocity, q3_velocity = np.nanpercentile(avg_velocity_masked.values, [25, 75])

    # Calculer les statistiques pour les amplitudes saisonnières
    median_amplitude = np.nanmedian(amplitude_masked.values)
    q1_amplitude, q3_amplitude = np.nanpercentile(amplitude_masked.values, [25, 75])

    # Calculer les statistiques pour les vitesses des pixels les plus rapides
    fastest_5_percent_threshold = np.nanpercentile(avg_velocity_masked.values, 95)
    mean_elevation_fastest = elevation.where(avg_velocity_masked > fastest_5_percent_threshold).mean().values

    # Calculer les statistiques pour la pente
    mean_slope = slope_masked.mean().values
    median_slope = np.nanmedian(slope_masked.values)
    q1_slope, q3_slope = np.nanpercentile(slope_masked.values, [25, 75])
    total_points = slope_masked.count().values  # Nombre total de points valides
    mask_low_slope = (slope_masked < 3)
    low_slope_points = slope_masked.where(mask_low_slope).count().values
    proportion_low_slope = (low_slope_points / total_points) * 100

    # Calculer les statistiques pour le débit
    avg_melt = melt_cycle_masked.mean(dim=["cycle"], skipna=True)
    mean_melt = avg_melt.mean().values
    median_melt = np.nanmedian(avg_melt.values)
    q1_melt, q3_melt = np.nanpercentile(avg_melt.values, [25, 75])

    # Afficher les statistiques
    print("Statistics for Annual Surface Velocities:")
    print(f"  - Mean velocity: {mean_velocity:.1f} m yr⁻¹")
    print(f"  - Median velocity: {median_velocity:.1f} m yr⁻¹")
    print(f"  - Interquartile range: {q1_velocity:.0f} to {q3_velocity:.0f} m yr⁻¹")
    print(f"  - Fastest 5% of pixels exceed: {fastest_5_percent_threshold:.1f} m yr⁻¹")
    print(f"  - Predominant elevation for fastest 5%: {mean_elevation_fastest:.0f} m\n")

    print("Statistics for Seasonal Velocity Variations:")
    print(f"  - Median amplitude: {median_amplitude:.1f} m yr⁻¹")
    print(f"  - Interquartile range for amplitude: {q1_amplitude:.0f} to {q3_amplitude:.0f} m yr⁻¹")

    print("Statistics for Slope:")
    print(f"  - Mean slope: {mean_slope:.1f} °")
    print(f"  - Median slope: {median_slope:.1f} °")
    print(f"  - Interquartile range: {q1_slope:.0f} to {q3_slope:.0f} °")
    print(f"  - Nombre total de points valides : {total_points}")
    print(f"  - Nombre de points avec pente < 3° : {low_slope_points}")
    print(f"  - Proportion de points avec pente < 3° : {proportion_low_slope:.2f}%")

    print("Statistics for Melt Rates:")
    print(f"  - Mean melt rate: {mean_melt:.3f} m w.e. day⁻¹")
    print(f"  - Median melt rate: {median_melt:.3f} m w.e. day⁻¹")
    print(f"  - Interquartile range: {q1_melt:.3f} to {q3_melt:.3f} m w.e. day⁻¹")

    # # Afficher les types des masks
    # print("Type de base_mask :", base_mask.dtype)
    # print("Type de result_mask :", result_mask.dtype)
    # print("Type de mask_xcount :", mask_xcount.dtype)
    # print("Type de mask_shadow :", mask_shadow.dtype)
    # print("Type de mask_snr :", mask_snr.dtype)

    # Liste des masques
    masks = [result_mask, mask_xcount, mask_shadow, mask_velavg, mask_snr]
    mask_names = ["Mask Total", "Mask XCount", "Mask Shadow", "Mask Velavg", "Mask SNR"]

    # Nombre de points valides vs total pour chaque masque
    for mask, name in zip(masks, mask_names):
        total_points = base_mask.sum().item()
        valid_points = mask.sum().item()

        print(f"{name}:")
        print(f"  - Nb of valid pts: {valid_points} / {total_points}")
        print(f"  - %age of valid pts: {(valid_points / total_points) * 100:.2f}%")
        print()


    mask_low = (slope < 12) & result_mask
    mask_high = (slope > 18) & result_mask

    from scipy.stats import pearsonr

    def compute_corr(mask):

        amp_vals = amplitude.where(mask).values.flatten()


        melt_vals = avg_melt_summer.where(mask).values.flatten()

        valid = np.isfinite(amp_vals) & np.isfinite(melt_vals)
        r, p = pearsonr(amp_vals[valid], melt_vals[valid])

        return r, p

    r_low, p_low = compute_corr(mask_low)
    r_high, p_high = compute_corr(mask_high)

    print("Low slopes (<12°): r =", r_low, "p =", p_low)
    print("High slopes (>18°): r =", r_high, "p =", p_high)


if __name__ == "__main__":
    # plot_random_timeseries(1, 12)
    # plot_2pixels_ts()
    # plot_histogram_extrema_slope()
    # plot_histogram_inflex_slope()
    # plot_daily_precip()
    # plot_typical_vel_melt_cycles()
    # plot_comp_vel_melt_cycles(0, 12, 18, 39)
    plot_main_figure_phasing_amplitude(0, 12, 18, 36, masks=True, stable_areas=False)
    # plot_map_serac_fall_slope_bins()
    # plot_map_slope_bins()
    # plot_hist_max()
    # plot_altitude_analysis()
    # plot_relative_amplitude_vs_slope()
    # plot_conceptual_effective_pressure_model()
    # plot_masks()
    # print_stats()
    # plot_friction_laws()
    # plot_CN_enveloppe()

