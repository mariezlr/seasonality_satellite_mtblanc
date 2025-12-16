from utils import *
from data_exploration import *
from main import *
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns



def plot_random_pixel_ts():
    # Trouver les indices des pixels valides
    notnan_pixels = vel_result.notnull().sum(dim='mid_date')
    valid_indices = np.argwhere(notnan_pixels.values >= 1)

    # Sélectionner le premier pixel valide (ou un aléatoire)
    y_idx, x_idx = valid_indices[0]

    # Extraire la série temporelle
    ts_orig_local = velocity.isel(y=y_idx, x=x_idx)
    ts_orig = vel_avg_detrended.isel(y=y_idx, x=x_idx)
    ts_filter = vel_lowpass.isel(y=y_idx, x=x_idx)

    fig = plt.figure(figsize=(10, 4))
    ts_orig_local.plot(label="Original", color="skyblue")
    ts_orig.plot(label="Spatial Avg", color="blue")
    ts_filter.plot(label="Filtered", color="red")

    plt.title(f'Série temporelle pour le pixel (y={y_idx}, x={x_idx})')
    plt.xlabel('Date')
    plt.ylabel('Vitesse')
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    fig.savefig(fig_dir / "random_pixel_ts.png")
    plt.close(fig)


def plot_validation_pixel_ts(x0, y0, df_data, name):

    # Extraire la série temporelle originale et filtrée à ce point avant filtrage
    ts_orig_local = velocity.sel(x=x0, y=y0, method='nearest')
    ts_orig = vel_avg_detrended.sel(x=x0, y=y0, method='nearest')
    ts_filter = vel_lowpass.sel(x=x0, y=y0, method='nearest')

    # Filter timeserie between 2016 and 2022
    t_min = ts_orig_local['mid_date'].min().values
    t_max = ts_orig_local['mid_date'].max().values

    temporal_mask = (df_data['date'] >= t_min) & (df_data['date'] <= t_max)
    data_sub = df_data[temporal_mask]

    # Plot
    fig = plt.figure(figsize=(10, 4))
    ts_orig_local.plot(label="Original", color="skyblue")
    ts_orig.plot(label="Spatial Avg", color="blue")
    ts_filter.plot(label="Filtered", color="red")
    plt.plot(data_sub['date'], data_sub['velocity'], color='green', label="obs sliding velocity")


    plt.title(f'Velocity timeseries for the nearest pixel from {name} (x={x0:.2f}, y={y0:.2f})')
    plt.xlabel('Date')
    plt.ylabel(r'Velocity (m yr$^{-1}$)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / f"validation_{name}.png")
    plt.close(fig)


def plot_annual_cycle_validation_point(df, name):
    df['day_of_year'] = df['date'].dt.dayofyear

    doy_mean = df.groupby('day_of_year')['velocity'].mean()

    fig, ax = plt.subplots(figsize=(11,5))
    ax.plot(doy_mean.index, doy_mean.values)
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Velocity (m/yr)")
    ax.set_title("Mean seasonal cycle of glacier sliding velocity")
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(fig_dir / f"annual_cycle_{name}.png")
    plt.close(fig)


def plot_xcount_ts_validation_points():
    fig, ax = plt.subplots()

    ts_xcount_arg4 = xcount.sel(x=x_utm_Arg4_GPS, y=y_utm_Arg4_GPS, method='nearest')
    ts_xcount_arg4.plot(ax=ax, color="green", label="Arg4")

    ts_xcount_argG = xcount.sel(x=x_utm_ArgG_GPS, y=y_utm_ArgG_GPS, method='nearest')
    ts_xcount_argG.plot(ax=ax, color="saddlebrown", label="ArgG")

    ts_xcount_argw = xcount.sel(x=x_utm_Argw, y=y_utm_Argw, method='nearest')
    ts_xcount_argw.plot(ax=ax, color="blue", label="Argwheel")

    ax.axhline(100, linestyle="--", color="k")
    ax.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / "arg_xcount.png")
    plt.close(fig)



def plot_mean_ts_all_pixels():
    fig = plt.figure()

    velocity.mean(dim=["x", "y"]).plot(label="Original mean vel", color="skyblue")
    vel_avg_detrended.mean(dim=["x", "y"]).plot(label="Spatial avg mean vel", color="blue")
    vel_lowpass.mean(dim=["x", "y"]).plot(label="Filtered mean vel", color="red")
    
    plt.xlabel("Date")
    plt.ylabel(r'Velocity (m yr$^{-1}$)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(fig_dir / "mean_ts_all_pixels.png")
    plt.close(fig)


def plot_random_pixels_avg_year():
    selected_pix = valid_indices[np.random.choice(len(valid_indices), 10, replace=False)]

    fig, ax = plt.subplots(figsize=(10,5))

    for i, (iy, ix) in enumerate(selected_pix):
        ax.plot(np.arange(1, 366), cycle_array[:, iy, ix], label=f"Pixel {i+1}")

    ax.set_xlabel("Day of year")
    ax.set_ylabel("Velocity (m yr$^{-1}$)")
    ax.set_title("Mean seasonal cycle (±2 days, cyclic) for 10 random pixels")
    ax.grid(True)
    ax.legend(ncol=2)

    plt.tight_layout()
    fig.savefig(fig_dir / "avg_vel_random_pixels.png")
    plt.close(fig)


def plot_random_timeseries(n_samples=6, seed=42):
    """Plots timeseries for N random valid pixels"""

    random.seed(seed)

    base_mask = xr.DataArray(result_mask.data.filled(0).astype(bool), dims=("y", "x"), 
                             coords={"y": velocity_cycle_mean["y"], "x": velocity_cycle_mean["x"]})

    print("Pixels valides après masque :", int(base_mask.sum()))

    # Valid pixels with non-NaN values
    valid_indices = []
    ny, nx = base_mask.sizes['y'], base_mask.sizes['x']

    for i in range(ny):
        for j in range(nx):
            if base_mask.values[i, j]:
                x0 = velocity_cycle_mean['x'].values[j]
                y0 = velocity_cycle_mean['y'].values[i]
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
    nrows, ncols = (2, 3) if n <= 6 else (3, 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7), sharex=True)
    axes = axes.flat

    lines_for_legend = []

    # Loop on pixels
    for k, (yi, xi) in enumerate(selected):
        ax = axes[k]

        x0 = velocity_cycle_mean['x'].values[xi]
        y0 = velocity_cycle_mean['y'].values[yi]

        ts_all_years = vel_result.sel(x=x0, y=y0)

        for j, year in enumerate(years):
            ts_year = ts_all_years.sel(mid_date=ts_all_years['mid_date.year'] == year)
            x_vals = ts_year['mid_date'].dt.dayofyear.astype(int).values
            y_vals = ts_year.values

            (line,) = ax.plot(x_vals, y_vals, color=cmap(j), alpha=0.7)

            if k == 0:  # keep lines for legend only once
                lines_for_legend.append((line, str(year)))

        # Mean seasonal vel
        ts_mean = velocity_cycle_mean.sel(x=x0, y=y0)
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
    plt.close(fig)




def plot_histogram_extrema_slope():

    sns.set_theme(style='whitegrid')

    # Prepare lists storing DOY values for each slope bin
    doymax_all = [[] for _ in range(n_bins)]
    doymin_all = [[] for _ in range(n_bins)]


    # Extract DOY values inside each slope interval
    for i in range(n_bins):
        mask = ((slope_arr >= slope_bins[i]) & (slope_arr < slope_bins[i+1]) & mask_arr)

        valmax = max_peak_doy[mask]
        valmax = valmax[~np.isnan(valmax)]
        doymax_all[i].extend(valmax)

        valmin = min_peak_doy[mask]
        valmin = valmin[~np.isnan(valmin)]
        doymin_all[i].extend(valmin)

    # Figure
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, wspace=0.12)

    # Panel (a) max
    sub_left = subfigs[0]
    axs_left = sub_left.subplots(4, 2)

    for i in range(n_bins):
        ax = axs_left[i // 2, i % 2]
        ax.hist(doymax_all[i], bins=month_bins, color='skyblue',
                edgecolor='k', alpha=0.8)
        ax.set_title(f"Slope {slope_bins[i]}–{slope_bins[i+1]}°",
                     fontsize=14)
        ax.set_xticks(mid_month_days)
        ax.set_xticklabels(month_labels_short)
        ax.grid(linestyle='--', alpha=0.4)

    axs_left[0,0].legend()
    sub_left.supxlabel("Month of maximal velocity", fontsize=20)
    sub_left.supylabel("Occurrences", fontsize=20)
    sub_left.text(0.01, 1, '(a)', fontsize=26, fontweight='bold', va='top')

    # Panel (b) min
    sub_right = subfigs[1]
    axs_right = sub_right.subplots(4, 2)

    for i in range(n_bins):
        ax = axs_right[i // 2, i % 2]
        ax.hist(doymin_all[i], bins=month_bins, color='violet',
                edgecolor='k', alpha=0.8)
        ax.set_title(f"Slope {slope_bins[i]}–{slope_bins[i+1]}°",
                     fontsize=14)
        ax.set_xticks(mid_month_days)
        ax.set_xticklabels(month_labels_short)
        ax.grid(linestyle='--', alpha=0.4)

    axs_right[0,0].legend()
    sub_right.supxlabel("Month of minimal velocity", fontsize=20)
    sub_right.supylabel("Occurrences", fontsize=20)
    sub_right.text(0.01, 1, '(b)', fontsize=26, fontweight='bold', va='top')

    fig.savefig(fig_dir / "doy_distrib_max_min.pdf", bbox_inches='tight')
    plt.close(fig)



def plot_histogram_inflex_slope():

    sns.set_theme(style='whitegrid')

    # Prepare lists storing DOY values for each slope bin
    doyinflex_all = [[] for _ in range(n_bins)]


    # Extract DOY values inside each slope interval
    for i in range(n_bins):
        mask = ((slope_arr >= slope_bins[i]) & (slope_arr < slope_bins[i+1]) & mask_arr)

        valinflex = max_peak_doy[mask]
        valinflex = valinflex[~np.isnan(valinflex)]
        doyinflex_all[i].extend(valinflex)

    # Figure
    fig, axes = plt.subplots(4, 2, figsize=(8, 10), constrained_layout=True)

    for i in range(n_bins):
        ax = axes[i // 2, i % 2]
        ax.hist(doyinflex_all[i], bins=month_bins, color='gold',
                edgecolor='k', alpha=0.8)
        ax.set_title(f"Slope {slope_bins[i]}–{slope_bins[i+1]}°",
                     fontsize=14)
        ax.set_xticks(mid_month_days)
        ax.set_xticklabels(month_labels_short)
        ax.grid(linestyle='--', alpha=0.4)

    axes[0,0].legend()

    fig.supxlabel("Month of main inflection point", fontsize=20)
    fig.supylabel("Occurrences", fontsize=20)

    fig.savefig(fig_dir / "doy_distrib_inflex.pdf", bbox_inches='tight')
    plt.close(fig)




def plot_meteofrance_map():
    fig, ax = plt.subplots(figsize=(10,8))

    # Fond satellite
    ax.imshow(np.moveaxis(img_map, 0, -1), extent=extent_map, origin='upper')

    x = np.arange(x_min_mf, x_max_mf, 1)
    y = [m*x+b for x in x]
    ax.plot(x,y, color = "red", label="Limite du massif")

    ax.scatter(df_mtblanc["x_utm"].values, df_mtblanc["y_utm"].values, color = "red", label="Stations Météo France")

    ax.set_xlim(np.nanmin(x_1d), np.nanmax(x_1d))
    ax.set_ylim(np.nanmin(y_1d), np.nanmax(y_1d))


    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=16)
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(fig_dir / "map_stations_meteo.png")
    plt.close(fig)



def plot_daily_precip():
    """
    Plot daily mean velocity for low- and high-slope areas,
    together with daily precipitation.
    Pixels outside the mask are fully excluded (NaN for all dates).
    """
    # Build 2D slope masks (numpy -> xarray)
    mask_flat_2d = (slope_arr < 10) & mask_arr
    mask_steep_2d = (slope_arr >= 15) & mask_arr

    print("Low slope pixels :", np.nansum(mask_flat_2d))
    print("High slope pixels:", np.nansum(mask_steep_2d))
    print("Overlap pixels   :", np.nansum(mask_flat_2d & mask_steep_2d))

    slope_masked = np.where(mask_arr, slope_arr, np.nan)
    mask_flat_2d  = slope_masked < 10
    mask_steep_2d = slope_masked >= 15

    print("Low slope pixels :", np.nansum(mask_flat_2d))
    print("High slope pixels:", np.nansum(mask_steep_2d))
    print("Overlap pixels   :", np.nansum(mask_flat_2d & mask_steep_2d))

    # Convert to xarray DataArray with spatial coordinates
    mask_flat = xr.DataArray(mask_flat_2d, dims=("y", "x"),
        coords={"y": vel_result.y, "x": vel_result.x})

    mask_steep = xr.DataArray(mask_steep_2d, dims=("y", "x"),
        coords={"y": vel_result.y, "x": vel_result.x})

    # Appliquer les masques sur velocity_filtered (données journalières)
    vel_low_daily = vel_result.where(mask_flat).mean(dim=["x","y"], skipna=True)
    vel_high_daily = vel_result.where(mask_steep).mean(dim=["x","y"], skipna=True)

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax1.plot(vel_low_daily['mid_date'], vel_low_daily, label="Slope < 10°", color='limegreen')
    ax1.plot(vel_high_daily['mid_date'], vel_high_daily, label="Slope ≥ 15°", color='crimson')
    ax1.set_xlabel("Date", fontsize=14)
    ax1.set_ylabel("Velocity (m/yr)", fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Axe y droit pour précipitations
    ax2 = ax1.twinx()
    ax2.plot(ts_daily.index, ts_daily.values, color='turquoise', alpha=0.7, label="Daily precip from MeteoFrance")
    ax2.plot(df_safran_precip['date'], df_safran_precip['precip'], color='blue', alpha=0.4, label="30-min precip from Safran")
    ax2.set_ylabel("Precipitation (mm)", color='blue', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper right')

    plt.title("Daily velocity & precipitation (2016-2022)")

    plt.tight_layout()
    fig.savefig(fig_dir / "daily_vel_precip.png")
    plt.close(fig)



def plot_typical_vel_melt_cycles():

    # Build 2D slope masks (numpy -> xarray)
    mask_flat_2d = (slope_arr < 10) & mask_arr
    mask_steep_2d = (slope_arr >= 15) & mask_arr

    # Convert to xarray DataArray with spatial coordinates
    mask_flat = xr.DataArray(mask_flat_2d, dims=("y", "x"),
        coords={"y": vel_result.y, "x": vel_result.x})

    mask_steep = xr.DataArray(mask_steep_2d, dims=("y", "x"),
        coords={"y": vel_result.y, "x": vel_result.x})

    # Masks applied to velocity cycles
    mean_vel_flat = velocity_cycle_mean.where(mask_flat).mean(dim=["x","y"], skipna=True)
    mean_vel_steep = velocity_cycle_mean.where(mask_steep).mean(dim=["x","y"], skipna=True)

    # Masks applied to melt rate cycles
    mean_melt_flat = melt_cycle_da.where(mask_flat).mean(dim=["x","y"], skipna=True)
    mean_melt_steep = melt_cycle_da.where(mask_steep).mean(dim=["x","y"], skipna=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(mean_vel_flat['doy_approx'], mean_vel_flat, label="Slope < 10°", color='limegreen')
    ax1.plot(mean_vel_steep['doy_approx'], mean_vel_steep, label=r"Slope $\geq$ 15°", color='crimson')
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
    plt.close(fig)


def plot_slope_map():
    # Définir les classes de pente (en degrés)
    slope_bins = np.arange(0, 65, 10)
    n_bins = len(slope_bins) - 1
    color_bins = ['#00ffff', '#3399ff', '#6666ff', '#66cc66', '#ffcc00', '#ff3300', '#cc0000']

    fig, ax = plt.subplots(figsize=(10,8))

    # Fond satellite
    ax.imshow(np.moveaxis(img_map, 0, -1), extent=extent_map, origin='upper')

    # Contours glaciers
    for i in mtblanc_outlines['geometry_id'].unique():
        contour = mtblanc_outlines[mtblanc_outlines['geometry_id'] == i]
        ax.plot(contour['x'], contour['y'], linewidth=1, color="grey")
        
    # Boucle sur chaque classe de pente et flatten des masques
    for i in range(n_bins):
        mask = ((slope_arr >= slope_bins[i]) & (slope_arr < slope_bins[i+1]) & mask_arr)
        mask_flat = mask.values.flatten()
        ax.scatter(x_1d[mask_flat], y_1d[mask_flat], 
                color=color_bins[i], s=0.2, label=f"{slope_bins[i]} $\leq$ slope < {slope_bins[i+1]}°", alpha = 0.8)

    ax.set_xlim(np.nanmin(x_1d), np.nanmax(x_1d))
    ax.set_ylim(np.nanmin(y_1d), np.nanmax(y_1d))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(markerscale=10)
    ax.grid(False)
    plt.show()

    plt.tight_layout()
    fig.savefig(fig_dir / "map_slope_bins.pdf")
    plt.close(fig)



def plot_altitude_analysis():

    z_vals = z.where(mask_arr).values.flatten()
    max_vals = max_peak_doy.where(mask_arr).values.flatten()
    min_vals = min_peak_doy.where(mask_arr).values.flatten()
    avg_vals = avg_velocity.where(mask_arr).values.flatten()
    ampl_vals = amplitude.where(mask_arr).values.flatten()

    # Moyenne glissante
    step = 10       # pas entre les centres (10 m)
    window = 200    # largeur de la fenêtre glissante (200 m)

    centers = np.arange(np.nanmin(z_vals), np.nanmax(z_vals)+step, step)

    centers, mean_maxdoy = moving_average(max_vals, z_vals, step, window)
    centers, mean_mindoy = moving_average(min_vals, z_vals, step, window)
    centers, mean_avg = moving_average(avg_vals, z_vals, step, window)
    centers, mean_ampl = moving_average(ampl_vals, z_vals, step, window)


    fig, ax1 = plt.subplots(figsize=(6,5))

    # Axe gauche : vitesse et amplitude
    ax1.plot(centers, mean_avg, color='limegreen', label='Average velocity')
    ax1.plot(centers, mean_ampl, color='darkorange', label='Amplitude')
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
    plt.show()


    fig.savefig("../output/figures/cycles_by_altitude.pdf")



if __name__ == "__main__":
    # plot_random_pixel_ts()
    # plot_validation_pixel_ts(x_utm_Arg4_GPS, y_utm_Arg4_GPS, data_GPS_ARG4, "Arg4")
    # plot_annual_cycle_validation_point(data_GPS_ARG4, "Arg4")
    # plot_validation_pixel_ts(x_utm_ArgG_GPS, y_utm_ArgG_GPS, data_GPS_ARGG, "ArgG")
    # plot_annual_cycle_validation_point(data_GPS_ARGG, "ArgG")
    # plot_validation_pixel_ts(x_utm_Argw, y_utm_Argw, data_Argwheel, "Arg wheel")
    # plot_annual_cycle_validation_point(data_Argwheel, "Arg wheel")
    plot_xcount_ts_validation_points()
    plot_mean_ts_all_pixels()
    plot_random_pixels_avg_year()
    plot_random_timeseries()
    plot_histogram_extrema_slope()
    plot_histogram_inflex_slope()
    plot_meteofrance_map()
    plot_daily_precip()
    plot_typical_vel_melt_cycles()
    plot_slope_map()
