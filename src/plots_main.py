from utils import *
from data_exploration import *
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

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
result_mask = ds_analysis["mask"]



def plot_random_timeseries(n_samples=6, seed=42):
    """Plots timeseries for N random valid pixels"""

    random.seed(seed)
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
    nrows, ncols = (2, 3) if n <= 6 else (3, 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7), sharex=True)
    axes = axes.flat

    lines_for_legend = []

    # Loop on pixels
    for k, (yi, xi) in enumerate(selected):
        ax = axes[k]

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
    plt.close(fig)


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
        ax.hist(doymax_all[i], bins=month_bins, color='skyblue',
                edgecolor='k', alpha=0.8)
        ax.set_title(f"Slope {slope_bins[i]}–{slope_bins[i+1]}°",
                     fontsize=14)
        ax.set_xticks(mid_month_days)
        ax.set_xticklabels(month_labels_short)
        ax.grid(linestyle='--', alpha=0.4)

#    axs_left[0,0].legend()
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

#    axs_right[0,0].legend()
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
                     fontsize=14)
        ax.set_xticks(mid_month_days)
        ax.set_xticklabels(month_labels_short)
        ax.grid(linestyle='--', alpha=0.4)

#    axes[0,0].legend()

    fig.supxlabel("Month of main inflection point", fontsize=20)
    fig.supylabel("Occurrences", fontsize=20)

    fig.savefig(fig_dir / "doy_distrib_inflex.pdf", bbox_inches='tight')
    plt.close(fig)



def plot_daily_precip():
    """
    Plot daily mean velocity for low- and high-slope areas,
    together with daily precipitation.
    Pixels outside the mask are fully excluded (NaN for all dates).
    """
    # Build 2D slope masks (numpy -> xarray)
    mask_flat_2d = (slope < 10) & result_mask
    mask_steep_2d = (slope >= 15) & result_mask
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


def plot_slope_repartition():
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
        mask = ((slope >= slope_bins[i]) & (slope < slope_bins[i+1]) & result_mask)
        mask_flat = mask.values.flatten()
        ax.scatter(x_1d[mask_flat], y_1d[mask_flat], 
                color=color_bins[i], s=0.2, label=f"{slope_bins[i]} $\leq$ slope < {slope_bins[i+1]}°", alpha = 0.8)

    ax.set_xlim(np.nanmin(x_1d), np.nanmax(x_1d))
    ax.set_ylim(np.nanmin(y_1d), np.nanmax(y_1d))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(markerscale=10, loc="upper left")
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(fig_dir / "map_slope_bins.pdf")
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
    plt.close(fig)



def plot_relative_amplitude_vs_slope():

    slope_vals = slope.where(result_mask).values.flatten()
    relampl_vals = amplitude_rel.where(result_mask).values.flatten()

    # Filter slopes <40°
    mask = (slope_vals >= 0) & (slope_vals <= 40)
    slope_vals_filtered = slope_vals[mask]
    relampl_vals_filtered = relampl_vals[mask]

    # Moyenne glissante
    step = 1       # pas entre les centres (10 m)
    window = 5    # largeur de la fenêtre glissante (200 m)

    centers, mean_relampl = moving_average(relampl_vals_filtered, slope_vals_filtered, step, window)
    

    fig, ax = plt.subplots(figsize=(6,5))

    # Axe gauche : vitesse et amplitude
    ax.plot(centers, mean_relampl, color='crimson', label='Relative amplitude')
    ax.set_xlabel('Slope (°)')
    ax.set_ylabel('Relative Amplitude')
    ax.grid(True, which='both', linestyle='--')

    # Légendes
    ax.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / "amplitude_vs_slope.pdf")
    plt.close(fig)



def plot_conceptual_effective_pressure_model():

    print(slope_min_intersect, slope_max_intersect)

    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(slope_line, tau_emp, linestyle='-', color='crimson', label='Average basal shear stress')

    ax.plot(slope_line, CN_min, linestyle='-', color='orange')
    ax.plot(slope_line, CN_max, linestyle='-', color='orange')
    ax.fill_between(slope_line, CN_min, CN_max,
        color="orange", alpha=0.3, label=r"$CN_{cavities}$ = $\tau_b(1/\theta)^{1/3}$ (in summer)")

    ax.plot(slope_line, CN_channels,
            color="blue", linestyle="--", label=r'$CN_{channels}$ = 0.29 tan$(\alpha)^{0.47}$ (in winter)')

    # Vertical lines at the two intersections
    plt.axvline(slope_min_intersect[-1], color='grey', linestyle=':', alpha=0.8)
    plt.axvline(slope_max_intersect[-1], color='grey', linestyle=':', alpha=0.8)


    # Shaded zone between the two intersections
    plt.axvspan(slope_min_intersect[-1], slope_max_intersect[-1], color='grey', alpha=0.3, hatch='//',
                label=f'Transition slope range : {slope_min_intersect[-1]:.2g}° - {slope_max_intersect[-1]:.2g}°')


    ax.set_xlabel("Surface slope (°)")
    ax.set_ylabel(fr"CN or $\tau_b$ (MPa)")

    ax.set_xscale("log")
    ax.set_yscale("log")

    x_ticks = np.arange(5, 45, 5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x:.0f}" for x in x_ticks])

    y_ticks = [0.05, 0.06, 0.10, 0.15, 0.20, 0.25]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.2f}" for y in y_ticks])

    ax.legend(loc = "upper left")
    ax.grid(linestyle="--")

    plt.tight_layout()
    fig.savefig(fig_dir / "N_vs_slope.pdf")
    plt.close(fig)

    


if __name__ == "__main__":
    plot_random_timeseries()
    print("plot_random_timeseries Done !")
    plot_histogram_inflex_slope()
    print("plot_histogram_inflex_slope Done !")
    plot_daily_precip()
    print("plot_daily_precip Done !")
    plot_typical_vel_melt_cycles()
    print("plot_typical_vel_melt_cycles Done !")
    plot_slope_repartition()
    print("plot_slope_repartition Done !")
    plot_altitude_analysis()
    print("plot_altitude_analysis Done !")
    plot_relative_amplitude_vs_slope()
    print("plot_relative_amplitude_vs_slope Done !")
    plot_conceptual_effective_pressure_model()
    print("plot_conceptual_effective_pressure_model Done !")
