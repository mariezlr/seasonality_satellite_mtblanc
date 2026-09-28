"""
Supplementary figures S1 to S6 of the manuscript.

Each figure is produced by one function below, which delegates to the
corresponding plotting routine in `plots_main.py` or `plots_annex.py`.
Running this file produces all six; passing figure numbers on the command
line produces only those.

    python plot_sup.py            # all
    python plot_sup.py 3 5        # only S3 and S5

Figure S7 (monthly analysis of raw velocity data) was produced by a
co-author with a processing script that is not part of this repository.
The resulting raster is provided under `data/` so that the figure can be
redrawn from it.

Outputs are written to `figures/paper/`.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import cm

# Project modules. These bring in the dataset variables (vel_result, slope,
# result_mask, vel_cycle, amplitude, elevation, base_mask, mask_xcount,
# mask_shadow, mask_velavg, mask_snr, x_1d, y_1d, mtblanc_outlines,
# df_safran_precip, glaciers) as well as out_dir, fig_dir, moving_average
# and the tau_b averaging parameters.
from utils import *
from data_exploration import *

from plots_config import FIG_DIR, load_dataset, save, GENTLE, STEEP

d = load_dataset()


def fig_s1():
    """S1 — Velocity time series at two representative pixels.

    Illustrates the raw and filtered signals behind the seasonal cycles
    used in the main analysis.
    """

    # Définir le rectangle de zoom
    xmin, xmax = 338500, 340500
    ymin, ymax = 5081500, 5083500

    candidate_low = (d.slope < 12) & d.result_mask
    candidate_high = (d.slope > 18) & d.result_mask

    y_target_low, x_target_low = 5083400, 340200
    y_target_high, x_target_high = 5082700, 339000 

    y_target_low, x_target_low = 5083400, 340200
    y_target_high, x_target_high = 5094750, 345050 

    # indices possibles selon le masque
    yi_possible_low, xi_possible_low = np.where(candidate_low.values)
    yi_possible_high, xi_possible_high = np.where(candidate_high.values)

    # distances aux coordonnées cibles
    dist_low = np.sqrt((d.velocity['x'].values[xi_possible_low] - x_target_low)**2 +
                    (d.velocity['y'].values[yi_possible_low] - y_target_low)**2)
    dist_high = np.sqrt((d.velocity['x'].values[xi_possible_high] - x_target_high)**2 +
                        (d.velocity['y'].values[yi_possible_high] - y_target_high)**2)

    # choisir le plus proche
    idx_low = dist_low.argmin()
    yi_low, xi_low = yi_possible_low[idx_low], xi_possible_low[idx_low]

    idx_high = dist_high.argmin()
    yi_high, xi_high = yi_possible_high[idx_high], xi_possible_high[idx_high]

    selected = [(yi_low, xi_low), (yi_high, xi_high)]
    labels = ["Slope < 12°", "Slope > 18°"]

    years = np.unique(d.velocity['mid_date.year'].values)
    cmap = plt.get_cmap("viridis", len(years))

    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    lines_for_legend = []

    for k, (yi, xi) in enumerate(selected):
        ax = axes[k]

        x0 = d.velocity['x'].values[xi]
        y0 = d.velocity['y'].values[yi]

        ts_all = d.velocity.sel(x=x0, y=y0)

        # Boucle sur les années
        for j, year in enumerate(years):
            ts_year = ts_all.sel(mid_date=ts_all['mid_date.year'] == year)
            x_vals = ts_year['mid_date'].dt.dayofyear.values
            y_vals = ts_year.values
            (line,) = ax.plot(x_vals, y_vals, color=cmap(j), alpha=0.7)
            if k == 0:
                lines_for_legend.append((line, str(year)))

        # Moyenne saisonnière
        ts_mean = d.vel_cycle.sel(x=x0, y=y0)
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
    
    # Ajout des labels (A) et (B)
    fig.text(0.01, 0.98, '(A)', fontsize=26, fontweight='bold', va='top')
    fig.text(0.51, 0.98, '(B)', fontsize=26, fontweight='bold', va='top')

    plt.tight_layout()
    save(fig, "figS1_2pixels_ts")
    plt.close(fig)


def fig_s2():
    """S2 — Daily velocity against daily precipitation.

    Plots daily mean velocity for low- and high-slope areas,
    together with daily precipitation.
    """
    # Build 2D slope masks (numpy -> xarray)
    mask_flat_2d = (d.slope <= 9) & d.result_mask
    mask_steep_2d = (d.slope >= 18) & d.result_mask
    print("Low slope pixels :", np.nansum(mask_flat_2d))
    print("High slope pixels:", np.nansum(mask_steep_2d))

    # Convert to xarray DataArray with spatial coordinates
    mask_flat = xr.DataArray(mask_flat_2d, dims=("y", "x"),
        coords={"y": d.velocity.y, "x": d.velocity.x})

    mask_steep = xr.DataArray(mask_steep_2d, dims=("y", "x"),
        coords={"y": d.velocity.y, "x": d.velocity.x})

    # Appliquer les masques sur velocity_filtered (données journalières)
    vel_low_daily = d.velocity.where(mask_flat).mean(dim=["x","y"], skipna=True)
    vel_high_daily = d.velocity.where(mask_steep).mean(dim=["x","y"], skipna=True)

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
    save(fig, "figS2_daily_vel_precip")
    plt.close(fig)


def fig_s3():
    """S3 — Basal shear stress against surface slope, all glaciers combined.

    Full Stokes estimates of tau_b used to constrain CN_cav in the
    conceptual model (Fig. 3).
    """
    all_slope = []
    all_tau_b = []

    for i, (glacier, info) in enumerate(glaciers.items()):
        df = pd.read_csv(out_dir / f"{glacier}.csv")

        mask = df["slope"] < 40        
        all_slope.append(df["slope"].where(mask).values.flatten())
        all_tau_b.append(df["tau_b"].where(mask).values.flatten())

    all_slope = np.concatenate(all_slope)
    all_tau_b = np.concatenate(all_tau_b)
        
    # Moyenne glissante
    centers, mean_tau_b = moving_average(all_tau_b, all_slope, step_taub_Elmer, window_taub_Elmer)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(all_slope, all_tau_b, color="grey", alpha=0.3, s=2, label="All points")
    ax.plot(centers, mean_tau_b, color="red", linewidth=2, label=fr"Mean $\tau_b$")

    ax.set_xlabel("Slope (°)", fontsize=14)
    ax.set_ylabel("Basal shear stress (MPa)", fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(markerscale=5, fontsize=10)

    plt.tight_layout()
    save(fig, "figS3_tau_b_by_slope_all_glaciers")
    plt.close(fig)

def fig_s4():
    """S4 — Map of the individual data-quality masks.

    Shows the spatial footprint of the shadow, SNR, count and mean-velocity
    masks.
    Convention: colored (semi-transparent) overlay = pixels EXCLUDED by
    that mask; transparent = pixels that PASS (valid).
    """
    from matplotlib.colors import LightSource
    from pyproj import Transformer

    transformer_to_ll = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

    def set_constant_lonlat_ticks(ax, lon_step, lat_step):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        corners = [(xlim[0], ylim[0]), (xlim[1], ylim[0]), (xlim[0], ylim[1]), (xlim[1], ylim[1])]
        corners_ll = [transformer_to_ll.transform(x, y) for x, y in corners]
        lons = [c[0] for c in corners_ll]
        lats = [c[1] for c in corners_ll]
        lon_c = 0.5 * (min(lons) + max(lons))
        lat_c = 0.5 * (min(lats) + max(lats))

        lon_ticks = np.arange(np.ceil(min(lons) / lon_step) * lon_step, max(lons), lon_step)
        xt, xtl = [], []
        for lon in lon_ticks:
            xu, _ = transformer_to_utm.transform(lon, lat_c)
            xt.append(xu); xtl.append(f"{lon:.2f}°E")

        lat_ticks = np.arange(np.ceil(min(lats) / lat_step) * lat_step, max(lats), lat_step)
        yt, ytl = [], []
        for lat in lat_ticks:
            _, yu = transformer_to_utm.transform(lon_c, lat)
            yt.append(yu); ytl.append(f"{lat:.2f}°N")

        ax.set_xticks(xt); ax.set_xticklabels(xtl, fontsize=11)
        ax.set_yticks(yt); ax.set_yticklabels(ytl, fontsize=11)

    # Shared hillshade background (more neutral than satellite at massif scale)
    dem_full = d.elevation.values
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(np.where(np.isfinite(dem_full), dem_full, 0),
                             vert_exag=2, dx=50, dy=50)
    extent = [float(d.elevation.x.min()), float(d.elevation.x.max()),
              float(d.elevation.y.min()), float(d.elevation.y.max())]

    xlim = (np.nanmin(x_1d)+9000, np.nanmax(x_1d)-8000)
    ylim = (np.nanmin(y_1d)+5000, np.nanmax(y_1d)-3000)


    masks_info = [
        (d.mask_xcount,  "Xcount",         "Number of valid observations\n$\\geq$ 80 obs. with xcount $\\geq$ 100",  '#e41a1c'),
        (d.mask_shadow,  "Shadow",          "Shadow fraction\n$<$ 50 days yr$^{-1}$",                                '#ff7f00'),
        (d.mask_velavg,  "Mean velocity",   "Mean surface velocity\n$\\geq$ 20 m yr$^{-1}$",                         '#984ea3'),
        (d.mask_snr,     "Seasonal SNR",    "Seasonal signal-to-noise ratio\n$\\geq$ 10",                            '#377eb8'),
    ]

    panel_labels = ['(A)', '(B)', '(C)', '(D)']

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for ax, (mask, short_title, criterion, color), label in zip(
        axes.flat, masks_info, panel_labels
    ):
        ax.imshow(hillshade, extent=extent, origin='lower', cmap='gray', vmin=0, vmax=1,
                 interpolation='bilinear')

        # glacier outlines for geographic context
        for gid in mtblanc_outlines['geometry_id'].unique():
            contour_g = mtblanc_outlines[mtblanc_outlines['geometry_id'] == gid]
            ax.plot(contour_g['x'], contour_g['y'], color='gray', linewidth=0.5, alpha=0.7)

        # CHANGED: overlay = EXCLUDED pixels (mask == False), not valid ones.
        # Use imshow rather than contourf to avoid alpha compositing artifacts.
        ny, nx = mask.values.shape
        rgba = np.zeros((ny, nx, 4), dtype=float)
        excluded = ~mask.values.astype(bool)
        import matplotlib.colors as mcolors
        rgb_excl = np.array(mcolors.to_rgb(color))
        rgba[excluded, :3] = rgb_excl
        rgba[excluded, 3] = 0.65  # semi-transparent

        ax.imshow(rgba,
                 extent=extent,
                 origin='lower', aspect='auto', zorder=3, interpolation='none')

        # Percentage passing this mask (for quantitative context)
        n_valid = int(mask.values.sum())
        n_total = int(d.base_mask.values.sum())
        pct = 100.0 * n_valid / n_total

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")

        set_constant_lonlat_ticks(ax, lon_step=0.1, lat_step=0.1)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.grid(False)

        # CHANGED: two-level title -- short name (bold) + criterion + % passing
        ax.set_title(f"{label} {short_title}",
                    fontsize=15, fontweight='bold', pad=6)
        print(f"{short_title}: {pct:.1f}%")

        # Small colored patch in the legend explaining the convention
        import matplotlib.patches as mpatches
        excl_patch = mpatches.Patch(color=color, alpha=0.65, label='Excluded')
        ax.legend(handles=[excl_patch], loc='lower right', fontsize=9,
                 framealpha=0.8)

    fig.tight_layout()
    save(fig, "figS4_masks_map")
    plt.close(fig)


def fig_s5():
    """S5 — Figure 2 recomputed without the quality masks.

    Same panels as Fig. 2 but using only the base mask, to verify that the
    slope dependence is not an artefact of the masking procedure.
    """
    from plot_fig2 import plot_main_figure_phasing_amplitude
    plot_main_figure_phasing_amplitude(
        GENTLE[0], GENTLE[1], STEEP[0], STEEP[1],
        masks=False, stable_areas=False,
    )


def fig_s6():
    """S6 — Seasonal amplitude and slope maps, masked areas only."""
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patheffects as pe
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from matplotlib.colors import LightSource
    from matplotlib import transforms
    from pyproj import Transformer
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature


    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

    # Rotation geometry
    valid_idx = d.base_mask.values.flatten().astype(bool)
    xs_valid = x_1d[valid_idx]
    ys_valid = y_1d[valid_idx]
    cx, cy = xs_valid.mean(), ys_valid.mean()

    def rotate_xy(xs, ys, cx, cy, angle_deg):
        theta = np.radians(angle_deg)
        xs_c, ys_c = xs - cx, ys - cy
        xs_rot = xs_c * np.cos(theta) - ys_c * np.sin(theta) + cx
        ys_rot = xs_c * np.sin(theta) + ys_c * np.cos(theta) + cy
        return xs_rot, ys_rot

    rotation_applied = 45
 
    xs_rot, ys_rot = rotate_xy(xs_valid, ys_valid, cx, cy, rotation_applied)
    pad_x = 0.01 * (xs_rot.max() - xs_rot.min())
    pad_y = 0.02 * (ys_rot.max() - ys_rot.min())
    xlim_rot = (xs_rot.min() - pad_x, xs_rot.max() + pad_x)
    ylim_rot = (ys_rot.min() - pad_y, ys_rot.max() + pad_y)
 

    fig = plt.figure(figsize=(9, 10))
    gs = gridspec.GridSpec(1, 2, wspace=0.18,
                           left=0.03, right=0.97, top=0.965, bottom=0.045)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]


    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    dem_full = d.elevation.values
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(np.where(np.isfinite(dem_full), dem_full, 0),
                             vert_exag=2, dx=50, dy=50)
    hs_extent = [float(d.elevation.x.min()), float(d.elevation.x.max()),
                 float(d.elevation.y.min()), float(d.elevation.y.max())]

    # Slope bin definitions (identical to plot_map_serac_fall_slope_bins)
    import matplotlib.colors as mcolors

    def sample_cmap_colors(cmap_name, n, vmin=0.25, vmax=0.85):
        cmap = plt.get_cmap(cmap_name)
        return [cmap(v) for v in np.linspace(vmin, vmax, n)]

    color_bins_1 = sample_cmap_colors('Blues', 3)
    color_bins_2 = sample_cmap_colors('YlOrBr', 3)
    transition_color = '#41ab5d'

    min_slope1, max_slope1, min_slope2, max_slope2 = 0, 12, 18, 36
    bins_low = np.linspace(min_slope1, max_slope1, 4)
    bins_high = np.linspace(min_slope2, max_slope2, 4)

    slope_bin_defs = []
    for i in range(3):
        slope_bin_defs.append((bins_low[i], bins_low[i+1], color_bins_1[i],
                               f"{bins_low[i]:.0f}°–{bins_low[i+1]:.0f}°"))
    slope_bin_defs.append((max_slope1, min_slope2, transition_color,
                           f"{max_slope1:.0f}°–{min_slope2:.0f}°"))
    for i in range(3):
        upper = bins_high[i+1] if i < 2 else np.inf
        label = (f"{bins_high[i]:.0f}°–{bins_high[i+1]:.0f}°" if i < 2
                 else f"$\\geq$ {bins_high[i]:.0f}°")
        slope_bin_defs.append((bins_high[i], upper, color_bins_2[i], label))

    def draw_map_base(ax, europe=True):
        """Draw hillshade + north arrow + Europe inset on ax."""
        rot_transform = (transforms.Affine2D().rotate_deg_around(cx, cy, rotation_applied)
                         + ax.transData)

        ax.imshow(hillshade, extent=hs_extent, origin='lower', cmap='gray', vmin=0, vmax=1,
                 interpolation='bilinear', transform=rot_transform, alpha=0.65)

        ax.set_xlim(xlim_rot); ax.set_ylim(ylim_rot)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # North arrow
        target_rot_x = xlim_rot[0] + 0.15 * (xlim_rot[1] - xlim_rot[0])
        target_rot_y = ylim_rot[0] + 0.94 * (ylim_rot[1] - ylim_rot[0])
        inv_theta = np.radians(-rotation_applied)
        tx, ty = target_rot_x - cx, target_rot_y - cy
        anchor_x = tx * np.cos(inv_theta) - ty * np.sin(inv_theta) + cx
        anchor_y = tx * np.sin(inv_theta) + ty * np.cos(inv_theta) + cy
        arrow_len_m = 0.08 * (ylim_rot[1] - ylim_rot[0])
        north_arrow = FancyArrowPatch(
            (anchor_x, anchor_y), (anchor_x, anchor_y + arrow_len_m),
            transform=rot_transform, arrowstyle='-|>', mutation_scale=20,
            color='black', linewidth=2.5, zorder=10)
        ax.add_patch(north_arrow)
        ax.text(anchor_x, anchor_y + arrow_len_m * 1.25, "N", transform=rot_transform,
               fontsize=20, fontweight='bold', ha='center', va='center', zorder=10)

        # Europe inset
        if europe:
            axins = ax.inset_axes([0.5, -0.1, 0.5, 0.32], transform=ax.transAxes,
                                projection=ccrs.PlateCarree())
            axins.set_extent([-10, 25, 35, 60], crs=ccrs.PlateCarree())
            axins.add_feature(cfeature.LAND, facecolor='whitesmoke', zorder=0)
            axins.add_feature(cfeature.OCEAN, facecolor='lightblue', zorder=0)
            axins.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='gray', zorder=1)
            axins.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=1)
            axins.plot(6.85, 45.85, marker='*', color='red', markersize=10,
                    markeredgecolor='black', markeredgewidth=0.5,
                    transform=ccrs.PlateCarree(), zorder=5)
            axins.set_xticks([]); axins.set_yticks([])
            for spine in axins.spines.values():
                spine.set_edgecolor('black'); spine.set_linewidth(0.8)

        return rot_transform

    # ------------------------------------------------------------------
    # Panel (A): amplitude map
    # ------------------------------------------------------------------
    ax_a = axes[0]
    rot_transform_a = draw_map_base(ax_a)

    amp_masked = d.amplitude.where(d.result_mask)
    finite_vals = amp_masked.values[np.isfinite(amp_masked.values)]
    vmin_a = np.percentile(finite_vals, 2)
    vmax_a = np.percentile(finite_vals, 98)

    # Use RGBA imshow (same approach as slope bins) to avoid aspect issues
    amp_norm = np.clip((amp_masked.values - vmin_a) / (vmax_a - vmin_a), 0, 1)
    viridis = plt.get_cmap('viridis')
    rgba_amp = viridis(amp_norm)
    rgba_amp[..., 3] = np.where(np.isfinite(amp_masked.values), 0.9, 0.0)

    ax_a.imshow(rgba_amp,
               extent=hs_extent, origin='lower', aspect='auto',
               zorder=3, interpolation='none', transform=rot_transform_a)


    # Glacier outlines -- already reprojected to UTM32632 in data_exploration.py
    for gid in mtblanc_outlines['geometry_id'].unique():
        contour = mtblanc_outlines[mtblanc_outlines['geometry_id'] == gid]
        ax_a.plot(contour['x'], contour['y'],
                color='black', linewidth=1.0, alpha=1.0, zorder=5,                   
                transform=rot_transform_a)
        
    # Colorbar
    import matplotlib.cm as cm
    sm = cm.ScalarMappable(cmap='viridis',
                           norm=plt.Normalize(vmin=vmin_a, vmax=vmax_a))
    sm.set_array([])
    cbar_a = fig.colorbar(sm, ax=ax_a, fraction=0.04, pad=0.02, shrink=0.7)
    cbar_a.set_label("Amplitude (m yr$^{-1}$)", fontsize=16)

    ax_a.set_xlim(xlim_rot)
    ax_a.set_ylim(ylim_rot)

    ax_a.set_title("(A)", fontsize=22, fontweight='bold')

    # ------------------------------------------------------------------
    # Panel (B): slope map
    # ------------------------------------------------------------------
    ax_b = axes[1]
    rot_transform_b = draw_map_base(ax_b, europe=False)

    legend_patches = []
    for lo, hi, col, lbl in slope_bin_defs:
        bmask = (d.slope >= lo) & (d.slope < hi) & d.result_mask
        bmask.plot.contourf(ax=ax_b, levels=[0.5, 1.5], colors=['none', col],
                            add_colorbar=False, transform=rot_transform_b)
        legend_patches.append(mpatches.Patch(color=col, label=lbl))


    # Glacier outlines -- already reprojected to UTM32632 in data_exploration.py
    for gid in mtblanc_outlines['geometry_id'].unique():
        contour = mtblanc_outlines[mtblanc_outlines['geometry_id'] == gid]
        ax_b.plot(contour['x'], contour['y'],
                color='black', linewidth=1.0, alpha=1.0, zorder=5,
                transform=rot_transform_b)
        
    ax_b.set_xlim(xlim_rot)
    ax_b.set_ylim(ylim_rot)
    ax_b.legend(handles=legend_patches, markerscale=2, fontsize=9,
               loc='upper right', title="Surface slope", title_fontsize=10)

    ax_b.set_title("(B)", fontsize=22, fontweight='bold')

    save(fig, "figS6_masked_amplitude_slope")
    plt.close(fig)


FIGURES = {1: fig_s1, 2: fig_s2, 3: fig_s3, 4: fig_s4, 5: fig_s5, 6: fig_s6}


def main(which=None):
    which = sorted(which) if which else sorted(FIGURES)
    print(f"Writing supplementary figures to {FIG_DIR}")
    for n in which:
        if n not in FIGURES:
            print(f"S{n}: no such figure, skipped")
            continue
        print(f"S{n} ...")
        FIGURES[n]()


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])