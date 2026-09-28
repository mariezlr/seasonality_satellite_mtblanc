"""
Figure 2 - Seasonal phasing and amplitude by surface-slope class.

    (A)      month of maximum velocity, by 6 degree slope bin
    (B, C)   mean seasonal velocity and melt cycles, gentle vs steep slopes
    (D)      relative seasonal amplitude against slope

The same function produces supplementary figure S5, with masks=False.

Output: figures/paper/fig2_phasing_amplitude.{pdf,png}

Run with:   python plot_fig2.py
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

# Project modules: month_bins, mid_month_days, month_labels_short and other
# shared globals used inside the plotting function.
from utils import *
from data_exploration import *

from plots_config import load_dataset, save, GENTLE, STEEP

d = load_dataset()


def plot_main_figure_phasing_amplitude(min_slope1, max_slope1, min_slope2, max_slope2, masks=True, stable_areas=False):

    from matplotlib.lines import Line2D

    fig_name = f"phasing_and_amplitude_{min_slope1}_{max_slope1}_{min_slope2}_{max_slope2}"

    if masks:
        result_mask = d.result_mask
    elif stable_areas:
        result_mask = d.base_mask & d.mask_stable_areas
        fig_name = f"phasing_and_amplitude_{min_slope1}_{max_slope1}_{min_slope2}_{max_slope2}_stable_areas"
    else:
        result_mask = d.base_mask
        fig_name = f"phasing_and_amplitude_nomask_{min_slope1}_{max_slope1}_{min_slope2}_{max_slope2}"

    # Colorblind-friendly light-to-dark color ramps (carried over from previous fix)
    def sample_cmap_colors(cmap_name, n, vmin=0.25, vmax=0.85):
        cmap = plt.get_cmap(cmap_name)
        return [cmap(v) for v in np.linspace(vmin, vmax, n)]

    color_bins_1 = sample_cmap_colors('Blues', 3)
    color_bins_2 = sample_cmap_colors('YlOrBr', 3)

    # helper to compute mean +/- SEM (std/sqrt(n)) in a moving
    # slope window -- SEM reflects confidence in the mean curve given the
    # number of pixels contributing to each window, rather than raw spatial
    # spread (which would be far larger and not the relevant uncertainty here).
    def moving_average_sem(y, x, step, window):
        centers = np.arange(np.nanmin(x), np.nanmax(x) + step, step)
        mean_result = np.full_like(centers, np.nan, dtype=float)
        sem_result = np.full_like(centers, np.nan, dtype=float)
        for i, c in enumerate(centers):
            sel = (x >= c - window / 2) & (x <= c + window / 2)
            n_sel = np.sum(sel)
            if n_sel > 0:
                mean_result[i] = np.nanmean(y[sel])
                sem_result[i] = np.nanstd(y[sel]) / np.sqrt(n_sel)
        return centers, mean_result, sem_result

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)

    # Create sub-figures
    subfigs_main = fig.subfigures(1, 2, width_ratios=[0.35, 0.65], wspace=0.08)
    sub_hist = subfigs_main[0]
    subfigs = subfigs_main[1].subfigures(3, 1, height_ratios=[0.34, 0.34, 0.32], hspace=0.08)

    ### (A) Amplitude vs slope
    slope_bins = np.arange(0, 37, 6)
    n_bins = len(slope_bins) - 1
    doymax_all = [[] for _ in range(n_bins)]
    doymin_all = [[] for _ in range(n_bins)]
    n_pixels_bin = []  # track sample size per slope bin

    for i in range(n_bins):
        mask = ((d.slope >= slope_bins[i]) & (d.slope < slope_bins[i+1]) & result_mask)

        valmax = d.max_peak_doy.where(mask)
        valmax = valmax.where(~np.isnan(valmax))
        doymax_all[i].extend(valmax.values.flatten())

        valmin = d.min_peak_doy.where(mask)
        valmin = valmin.where(~np.isnan(valmin))
        doymin_all[i].extend(valmin.values.flatten())

        n_pixels_bin.append(int(mask.sum()))  # IMPROVEMENT 5

    axs_hist = sub_hist.subplots(6, 1, sharex=True)  # IMPROVEMENT 4: shared x-axis

    for i in range(n_bins):
        ax = axs_hist[i]
        ax.hist(doymax_all[i], bins=month_bins, color="#008080",
                edgecolor='k', alpha=0.6)
        # sample size annotation
        ax.set_title(f"Slope {slope_bins[i]}–{slope_bins[i+1]}° (n={n_pixels_bin[i]})",
                     fontsize=14)
        ax.set_xticks(mid_month_days)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(linestyle='--', alpha=0.4)
        # hide x tick labels on all but the bottom subplot
        if i < n_bins - 1:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xticklabels(month_labels_short, fontsize=14)

    sub_hist.supxlabel("Month of maximal velocity", fontsize=18)
    sub_hist.supylabel("Occurrences", fontsize=18)


    # Rescaling between 0 and 1 for each point (x, y)
    vel_avg = d.vel_cycle.mean(dim="cycle")
    vel_rescaled = d.vel_cycle / vel_avg

    ### (B) & (C) seasonal velocity cycles
    ax_cycle1 = subfigs[0].subplots()
    ax_cycle2 = subfigs[1].subplots()

    for i, ax in enumerate([ax_cycle1, ax_cycle2]):

        if i == 0:
            color_bins = color_bins_1
            min_slope, max_slope = min_slope1, max_slope1
            mask_all_2d = (d.slope >= min_slope1) & (d.slope < min_slope2) & result_mask
            label_melt = fr" Mean melt rate: {min_slope1:.0f}° $\leq$ slope < {min_slope2:.0f}°"


        else:
            color_bins = color_bins_2
            min_slope, max_slope = min_slope2, max_slope2
            mask_all_2d = (d.slope >= max_slope1) & (d.slope < max_slope2) & result_mask
            label_melt = fr" Mean melt rate: {max_slope1:.0f}° $\leq$ slope < {max_slope2:.0f}°"



        bins_temp = np.linspace(min_slope, max_slope, 4)

        mask_flat_2d = (d.slope >= bins_temp[0]) & (d.slope < bins_temp[1]) & result_mask
        mask_mid_2d = (d.slope >= bins_temp[1]) & (d.slope < bins_temp[2]) & result_mask
        mask_steep_2d = (d.slope >= bins_temp[2]) & (d.slope < bins_temp[3]) & result_mask
        mask_transition_2d = (d.slope >= max_slope1) & (d.slope < min_slope2) & result_mask

        mask_flat = xr.DataArray(mask_flat_2d, dims=("y", "x"),
            coords={"y": d.velocity.y, "x": d.velocity.x})
        mask_mid = xr.DataArray(mask_mid_2d, dims=("y", "x"),
            coords={"y": d.velocity.y, "x": d.velocity.x})
        mask_steep = xr.DataArray(mask_steep_2d, dims=("y", "x"),
            coords={"y": d.velocity.y, "x": d.velocity.x})
        mask_all = xr.DataArray(mask_all_2d, dims=("y", "x"),
            coords={"y": d.velocity.y, "x": d.velocity.x})
        mask_trans = xr.DataArray(mask_transition_2d, dims=("y", "x"),
            coords={"y": d.velocity.y, "x": d.velocity.x})

        # sample size per slope class (for legend labels)
        n_flat = int(mask_flat.sum())
        n_mid = int(mask_mid.sum())
        n_steep = int(mask_steep.sum())
        n_trans = int(mask_trans.sum())

        # Mean + SEM (standard error of the mean = std/sqrt(n), IMPROVEMENT 3)
        # rather than raw std: we are showing confidence in the MEAN curve,
        # not the full spatial spread across pixels (which is much larger
        # and not the relevant uncertainty here).
        mean_vel_flat = vel_rescaled.where(mask_flat).mean(dim=["x","y"], skipna=True)[5:-5]
        std_vel_flat  = vel_rescaled.where(mask_flat).std(dim=["x","y"], skipna=True)[5:-5] / np.sqrt(max(n_flat, 1))
        mean_vel_mid = vel_rescaled.where(mask_mid).mean(dim=["x","y"], skipna=True)[5:-5]
        std_vel_mid  = vel_rescaled.where(mask_mid).std(dim=["x","y"], skipna=True)[5:-5] / np.sqrt(max(n_mid, 1))
        mean_vel_steep = vel_rescaled.where(mask_steep).mean(dim=["x","y"], skipna=True)[5:-5]
        std_vel_steep  = vel_rescaled.where(mask_steep).std(dim=["x","y"], skipna=True)[5:-5] / np.sqrt(max(n_steep, 1))
        mean_vel_trans = vel_rescaled.where(mask_trans).mean(dim=["x","y"], skipna=True)[5:-5]
        std_vel_trans  = vel_rescaled.where(mask_trans).std(dim=["x","y"], skipna=True)[5:-5] / np.sqrt(max(n_trans, 1))

        mean_melt_all = d.melt_cycle.where(mask_all).mean(dim=["x","y"], skipna=True)

        # append sample size to legend labels
        label_trans = fr" {max_slope1:.0f}° $\leq$ slope < {min_slope2:.0f}° (n={n_trans})"
        label_flat  = fr" {bins_temp[0]:.0f}° $\leq$ slope < {bins_temp[1]:.0f}° (n={n_flat})"
        label_mid   = fr" {bins_temp[1]:.0f}° $\leq$ slope < {bins_temp[2]:.0f}° (n={n_mid})"
        label_steep = fr" {bins_temp[2]:.0f}° $\leq$ slope < {bins_temp[3]:.0f}° (n={n_steep})"

        if i == 1:
            ax.plot(mean_vel_trans['doy_approx'], mean_vel_trans, color="k", alpha=0.8, label=label_trans)
            ax.fill_between(mean_vel_trans['doy_approx'], mean_vel_trans - std_vel_trans,
                            mean_vel_trans + std_vel_trans, color="k", alpha=0.10)

        for mean_v, std_v, lbl, col in zip(
            [mean_vel_flat, mean_vel_mid, mean_vel_steep],
            [std_vel_flat, std_vel_mid, std_vel_steep],
            [label_flat, label_mid, label_steep],
            color_bins,
        ):
            ax.plot(mean_v['doy_approx'], mean_v, color=col, label=lbl)
            # +/- 1 std shaded band around the mean velocity curve
            ax.fill_between(mean_v['doy_approx'], mean_v - std_v, mean_v + std_v,
                            color=col, alpha=0.15)

        if i == 0:
            ax.plot(mean_vel_trans['doy_approx'], mean_vel_trans, color="k", alpha=0.8, label=label_trans)
            ax.fill_between(mean_vel_trans['doy_approx'], mean_vel_trans - std_vel_trans,
                            mean_vel_trans + std_vel_trans, color="k", alpha=0.10)

        ax.set_ylim(0.85, 1.15)
        ax.set_xlabel("Day of year", fontsize=18)
        ax.set_ylabel(r"Normalized Velocity", color='black', fontsize=18)
        ax.tick_params(axis='y', labelcolor='black')
        ax.grid(True)

        ax2 = ax.twinx()
        ax2.plot(mean_melt_all['doy_approx'], mean_melt_all, color="purple", linestyle="--", alpha=0.7, label=label_melt)
        ax2.set_ylabel(r"Melt rate (m w.e. day$^{-1}$)", color='purple', fontsize=18)
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.set_ylim(0, 0.07)
        ax2.grid(True, which='both', axis='y', linestyle='--', color='purple', alpha=0.3)

        # merged legend (solid = velocity, dashed = melt),
        legend_loc = 'upper right' if i == 0 else 'upper left'
        ax.legend(handles=ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0],
                  labels=ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1], 
                  loc=legend_loc, fontsize=8, ncol=1)

        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)


    ### (D) Amplitude vs slope
    slope_vals = d.slope.where(result_mask).values.flatten()
    relampl_vals = d.amplitude_rel.where(result_mask).values.flatten()
    melt_vals = d.avg_melt_summer.where(result_mask).values.flatten()

    mask = (slope_vals >= 0) & (slope_vals <= 36)
    slope_vals_filtered = slope_vals[mask]
    relampl_vals_filtered = relampl_vals[mask]
    melt_vals_filtered = melt_vals[mask]

    step = 1
    window = 3

    # mean +/- SEM bands for panel (D) too
    centers, mean_relampl, std_relampl = moving_average_sem(relampl_vals_filtered, slope_vals_filtered, step, window)
    centers, mean_melt, std_melt = moving_average_sem(melt_vals_filtered, slope_vals_filtered, step, window)

    ax_amp = subfigs[2].subplots()

    ax_amp.plot(centers, mean_relampl, color='green', label='Relative amplitude')
    ax_amp.fill_between(centers, mean_relampl - std_relampl, mean_relampl + std_relampl,
                        color='green', alpha=0.15)
    ax_amp.set_xlabel('Slope (°)', fontsize=18)
    ax_amp.set_ylabel('Relative Amplitude', color="green", fontsize=18)
    ax_amp.tick_params(axis='y', labelcolor='green')
    ax_amp.grid(True, which='both', linestyle='--', color='green', alpha=0.3)

    # ax4 = ax_amp.twinx()
    # ax4.plot(centers, mean_melt, color='purple', linestyle="--", label='Summer melt')
    # ax4.fill_between(centers, mean_melt - std_melt, mean_melt + std_melt,
    #                  color='purple', alpha=0.15)
    # ax4.set_ylabel(r"Mean summer melt rate" "\n" r"(m w.e. day$^{-1}$)", color='purple', fontsize=18)
    # ax4.tick_params(axis='y', labelcolor='purple')
    # ax4.grid(True, which='both', axis='y', linestyle='--', color='purple', alpha=0.3)

    ax_amp.legend(loc='upper left', fontsize=12)


    # Panel labels
    fig.text(0.01, 1.02, '(A)', fontsize=26, fontweight='bold', va='top')
    fig.text(0.35, 1.02, '(B)', fontsize=26, fontweight='bold', va='top')
    fig.text(0.35, 0.68, '(C)', fontsize=26, fontweight='bold', va='top')
    fig.text(0.35, 0.34, '(D)', fontsize=26, fontweight='bold', va='top')

    save(fig, f"fig2_{fig_name}")
    plt.close(fig)

if __name__ == "__main__":
    plot_main_figure_phasing_amplitude(GENTLE[0], GENTLE[1], STEEP[0], STEEP[1], masks=True, stable_areas=False)