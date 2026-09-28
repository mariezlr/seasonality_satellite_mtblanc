"""
Figure 1 - Slope breaks and seasonal phasing of surface velocity.

    (A)           map of surface-slope classes over the Mont Blanc massif,
                  coloured by the day of maximum velocity
    (B, C, D, E)  profiles along flow lines crossing four slope breaks

Output: figures/paper/fig1_slope_breaks.{pdf,png,svg}

Run with:   python plot_fig1.py
"""

from pathlib import Path

import cmocean
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import transforms
from matplotlib.colors import LightSource
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, uniform_filter1d

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Project modules: x_1d, y_1d, mtblanc_outlines, out_dir and other shared
# globals used inside the plotting function.
from utils import *
from data_exploration import *

from plots_config import ROSE_PATH, load_dataset, save, GENTLE, STEEP

d = load_dataset()


def plot_slope_breaks_map_and_profiles(
        elevation, slope, max_peak_doy, base_mask,
        shadow_mask=None,
        min_slope1=GENTLE[0], max_slope1=GENTLE[1], min_slope2=STEEP[0], max_slope2=STEEP[1],
        candidates=None, half_width_m=800,
        flow_length_m=1200, n_flow_points=120,
        rotation_applied=45, scalebar_m=5000, slope_alpha=0.75,
        shadow_alpha=0.28, rose_path=ROSE_PATH):

    if candidates is None:
        candidates = [
            (6.982, 45.959),
            (7.033, 45.917),
            (6.932, 45.878),
            (6.812, 45.847),
        ]
    n_candidates = len(candidates)

    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    cyclic_cmap = cmocean.cm.phase

    # ------------------------------------------------------------------
    # Orientation des rasters
    # ------------------------------------------------------------------
    # origin='lower' suppose des y croissants. Un DEM stocke nord-en-haut a
    # des y decroissants : sans retournement l'eclairage vient du mauvais cote
    # et le relief apparait "en creux".
    def ensure_ascending_y(da):
        yv = da["y"].values
        if yv.size > 1 and yv[1] < yv[0]:
            return da.isel(y=slice(None, None, -1))
        return da

    elevation = ensure_ascending_y(elevation)
    slope = ensure_ascending_y(slope)
    max_peak_doy = ensure_ascending_y(max_peak_doy)
    base_mask = ensure_ascending_y(base_mask)
    if shadow_mask is not None:
        shadow_mask = ensure_ascending_y(shadow_mask)

    x_vals = elevation.x.values
    y_vals = elevation.y.values
    dx_m = float(abs(x_vals[1] - x_vals[0]))
    dy_m = float(abs(y_vals[1] - y_vals[0]))

    # extent sur les BORDS de pixels (sinon decalage d'un demi-pixel)
    extent = [x_vals.min() - dx_m / 2, x_vals.max() + dx_m / 2,
              y_vals.min() - dy_m / 2, y_vals.max() + dy_m / 2]

    # 'nearest' + stage 'data' : le reechantillonnage en espace ecran apres la
    # rotation affine produit un moire diagonal caracteristique.
    im_kw = dict(origin="lower", interpolation="nearest",
                 interpolation_stage="data", transform=None)

    # ------------------------------------------------------------------
    # Remplissage des NaN par plus proche voisin
    # ------------------------------------------------------------------
    dem_raw = elevation.values.astype(float)
    finite = np.isfinite(dem_raw)
    if finite.all():
        dem_filled = dem_raw
    else:
        idx = distance_transform_edt(~finite, return_distances=False,
                                     return_indices=True)
        dem_filled = dem_raw[tuple(idx)]

    # ------------------------------------------------------------------
    # Interpolateurs pour le trace des lignes d'ecoulement
    # ------------------------------------------------------------------
    dz_dy, dz_dx = np.gradient(dem_filled, dy_m, dx_m)
    grad_mag = np.hypot(dz_dx, dz_dy)
    grad_mag_safe = np.where(grad_mag > 1e-6, grad_mag, np.nan)

    common_kw = dict(bounds_error=False, fill_value=np.nan)
    ux_interp = RegularGridInterpolator((y_vals, x_vals), -dz_dx / grad_mag_safe, **common_kw)
    uy_interp = RegularGridInterpolator((y_vals, x_vals), -dz_dy / grad_mag_safe, **common_kw)
    elev_interp = RegularGridInterpolator((y_vals, x_vals),
                                          np.where(finite, dem_filled, np.nan), **common_kw)
    doy_interp = RegularGridInterpolator((y_vals, x_vals), max_peak_doy.values, **common_kw)

    def trace_flowline(cx_utm, cy_utm):
        def interp_vec(px, py):
            ux = ux_interp([[py, px]])[0]
            uy = uy_interp([[py, px]])[0]
            norm = np.hypot(ux, uy)
            if not np.isfinite(norm) or norm < 1e-6:
                return 0.0, 0.0
            return float(ux / norm), float(uy / norm)

        step = flow_length_m / n_flow_points
        down_pts = [(cx_utm, cy_utm)]
        px, py = cx_utm, cy_utm
        for _ in range(n_flow_points // 2):
            ux, uy = interp_vec(px, py)
            px, py = px + step * ux, py + step * uy
            down_pts.append((px, py))
        up_pts = []
        px, py = cx_utm, cy_utm
        for _ in range(n_flow_points // 2):
            ux, uy = interp_vec(px, py)
            px, py = px - step * ux, py - step * uy
            up_pts.append((px, py))
        flow_pts = list(reversed(up_pts)) + down_pts
        xf = np.array([p[0] for p in flow_pts])
        yf = np.array([p[1] for p in flow_pts])
        d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xf), np.diff(yf)))])
        return xf, yf, d

    # ------------------------------------------------------------------
    # Bins de pente
    # ------------------------------------------------------------------
    def sample_cmap_colors(cmap_name, n, vmin=0.25, vmax=0.85):
        cmap = plt.get_cmap(cmap_name)
        return [cmap(v) for v in np.linspace(vmin, vmax, n)]

    color_bins_1 = sample_cmap_colors('Blues', 3)
    color_bins_2 = sample_cmap_colors('YlOrBr', 3)
    transition_color = '#66c2a4'

    bins_low = np.linspace(min_slope1, max_slope1, 4)
    bins_high = np.linspace(min_slope2, max_slope2, 4)

    slope_bin_defs = []
    for i in range(3):
        slope_bin_defs.append((bins_low[i], bins_low[i + 1], color_bins_1[i],
                               f"{bins_low[i]:.0f}\u00b0\u2013{bins_low[i + 1]:.0f}\u00b0"))
    slope_bin_defs.append((max_slope1, min_slope2, transition_color,
                           f"{max_slope1:.0f}\u00b0\u2013{min_slope2:.0f}\u00b0"))
    for i in range(3):
        upper = bins_high[i + 1] if i < 2 else np.inf
        label = (f"{bins_high[i]:.0f}\u00b0\u2013{bins_high[i + 1]:.0f}\u00b0" if i < 2
                 else f"$\\geq$ {bins_high[i]:.0f}\u00b0")
        slope_bin_defs.append((bins_high[i], upper, color_bins_2[i], label))

    # ------------------------------------------------------------------
    # Geometrie de la figure
    # ------------------------------------------------------------------
    def rotate_xy(xs, ys, cx, cy, angle_deg):
        theta = np.radians(angle_deg)
        xs_c, ys_c = np.asarray(xs) - cx, np.asarray(ys) - cy
        xs_rot = xs_c * np.cos(theta) - ys_c * np.sin(theta) + cx
        ys_rot = xs_c * np.sin(theta) + ys_c * np.cos(theta) + cy
        return xs_rot, ys_rot

    x_2d, y_2d = np.meshgrid(x_vals, y_vals)
    valid = base_mask.values.astype(bool)
    xs_valid, ys_valid = x_2d[valid], y_2d[valid]
    cx, cy = xs_valid.mean(), ys_valid.mean()

    xs_rot, ys_rot = rotate_xy(xs_valid, ys_valid, cx, cy, rotation_applied)
    pad_x = 0.01 * (xs_rot.max() - xs_rot.min())
    pad_y = 0.02 * (ys_rot.max() - ys_rot.min())
    xlim_rot = (xs_rot.min() - pad_x, xs_rot.max() + pad_x)
    ylim_rot = (ys_rot.min() - pad_y, ys_rot.max() + pad_y)

    aspect_data = (ylim_rot[1] - ylim_rot[0]) / (xlim_rot[1] - xlim_rot[0])
    fig_height_in = 12.0
    top_frac, bottom_frac = 0.97, 0.04
    map_width_in = fig_height_in * (top_frac - bottom_frac) / aspect_data
    grid_width_in = 10.0
    fig_width_in = map_width_in + grid_width_in + 0.4

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    gs_main = gridspec.GridSpec(1, 2, width_ratios=[map_width_in, grid_width_in],
                                wspace=0.2, left=0.01, right=0.99,
                                top=top_frac, bottom=bottom_frac)
    ax_map = fig.add_subplot(gs_main[0])

    gs_right_outer = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_main[1],
        height_ratios=[0.08, 0.92, 0.08], hspace=0)
    n_cols = 2 if n_candidates > 1 else 1
    n_rows = int(np.ceil(n_candidates / n_cols))
    gs_grid = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=gs_right_outer[1],
        hspace=0.65, wspace=0.55)
    profile_axes = [fig.add_subplot(gs_grid[i // n_cols, i % n_cols])
                    for i in range(n_candidates)]

    # ------------------------------------------------------------------
    # PANEL A : hillshade
    # ------------------------------------------------------------------
    rot_transform = (transforms.Affine2D().rotate_deg_around(cx, cy, rotation_applied)
                     + ax_map.transData)
    im_kw["transform"] = rot_transform

    ls_hill = LightSource(azdeg=315, altdeg=45)
    hs = ls_hill.hillshade(dem_filled, vert_exag=2, dx=dx_m, dy=dy_m)
    hs_rgba = plt.get_cmap('gray')(np.clip(hs, 0, 1))
    hs_rgba[..., 3] = finite.astype(float)      # transparent hors DEM
    ax_map.imshow(hs_rgba, extent=extent, zorder=0, **im_kw)

    # ------------------------------------------------------------------
    # PANEL A : bins de pente (imshow categoriel)
    # ------------------------------------------------------------------
    slope_vals = slope.values
    bin_id = np.full(slope_vals.shape, np.nan)
    for k, (lo, hi, col, lbl) in enumerate(slope_bin_defs):
        bin_id[(slope_vals >= lo) & (slope_vals < hi) & valid] = k

    bins_rgba = np.zeros(bin_id.shape + (4,))
    for k, (lo, hi, col, lbl) in enumerate(slope_bin_defs):
        sel = bin_id == k
        if sel.any():
            bins_rgba[sel] = mcolors.to_rgba(col)
    bins_rgba[..., 3] = np.where(np.isfinite(bin_id), slope_alpha, 0.0)
    ax_map.imshow(bins_rgba, extent=extent, zorder=1, **im_kw)

    legend_patches = [mpatches.Patch(color=d[2], label=d[3]) for d in slope_bin_defs]

    # ------------------------------------------------------------------
    # PANEL A : masque d'ombre
    # ------------------------------------------------------------------
    shadow_vals = None
    if shadow_mask is not None:
        shadow_vals = np.nan_to_num(shadow_mask.values.astype(float))
        sh_rgba = np.zeros(shadow_vals.shape + (4,))
        sh_rgba[..., :3] = 0.25
        sh_rgba[..., 3] = np.where(shadow_vals > 0.5, shadow_alpha, 0.0)
        ax_map.imshow(sh_rgba, extent=extent, zorder=4, **im_kw)
        legend_patches.append(mpatches.Patch(facecolor='0.25', alpha=shadow_alpha,
                                             label="Terrain shadow mask"))

    # limites fixees APRES tous les traces
    ax_map.set_xlim(xlim_rot)
    ax_map.set_ylim(ylim_rot)
    ax_map.set_aspect("equal")
    ax_map.set_xticks([]); ax_map.set_yticks([])
    ax_map.set_xlabel(''); ax_map.set_ylabel('')
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    ax_map.set_title("(A)", fontsize=22, fontweight="bold")
    ax_map.legend(handles=legend_patches, loc="upper left", fontsize=10,
                  framealpha=0.9, title="Slope", title_fontsize=12)

    # ------------------------------------------------------------------
    # Barre d'echelle (rotation rigide -> longueurs exactes)
    # ------------------------------------------------------------------
    span = xlim_rot[1] - xlim_rot[0]
    frac = scalebar_m / span
    x0, y0 = 0.06, 0.06
    ax_map.plot([x0, x0 + frac], [y0, y0], transform=ax_map.transAxes,
                color='k', lw=3, solid_capstyle='butt', zorder=10,
                path_effects=[pe.Stroke(linewidth=5.5, foreground='white'), pe.Normal()])
    ax_map.text(x0 + frac / 2, y0 + 0.012, f"{scalebar_m / 1000:.0f} km",
                transform=ax_map.transAxes, ha='center', va='bottom',
                fontsize=12, fontweight="bold", zorder=10,
                path_effects=[pe.Stroke(linewidth=2.5, foreground='white'), pe.Normal()])

    # ------------------------------------------------------------------
    # Rose des vents
    # ------------------------------------------------------------------
    rose_path = Path(rose_path)
    if rose_path.is_file():
        rose_rotated = Image.open(rose_path).convert("RGBA").rotate(
            rotation_applied, expand=True, resample=Image.BICUBIC)
        bbox_crop = rose_rotated.getbbox()      # rogne les marges transparentes
        if bbox_crop:
            rose_rotated = rose_rotated.crop(bbox_crop)
        ab = AnnotationBbox(OffsetImage(np.array(rose_rotated), zoom=0.75),
                            (0.85, 0.94), xycoords='axes fraction',
                            frameon=True, pad=0.01, box_alignment=(0.5, 0.5))
        ax_map.add_artist(ab)
    else:
        print(f"[warn] rose des vents introuvable : {rose_path}")

    # ------------------------------------------------------------------
    # Inset Europe
    # ------------------------------------------------------------------
    axins = ax_map.inset_axes([0.5, -0.08, 0.5, 0.28], transform=ax_map.transAxes,
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

    # ------------------------------------------------------------------
    # Ordre des sites du nord au sud sur la carte tournee
    # ------------------------------------------------------------------
    cand_utm = [transformer_to_utm.transform(lon, lat) for lon, lat in candidates]
    cand_rot_y = [rotate_xy(np.array([ux]), np.array([uy]), cx, cy, rotation_applied)[1][0]
                  for ux, uy in cand_utm]
    order = np.argsort(cand_rot_y)[::-1]
    letters = [chr(ord('B') + i) for i in range(n_candidates)]
    shadow_fractions = {}

    for rank, idx in enumerate(order):
        letter = letters[rank]
        lon, lat = candidates[idx]
        ux, uy = cand_utm[idx]

        # boite noire a halo blanc
        box_x = [ux - half_width_m, ux + half_width_m,
                 ux + half_width_m, ux - half_width_m, ux - half_width_m]
        box_y = [uy - half_width_m, uy - half_width_m,
                 uy + half_width_m, uy + half_width_m, uy - half_width_m]
        ax_map.plot(box_x, box_y, color='white', linewidth=5.5,
                    transform=rot_transform, zorder=7)
        ax_map.plot(box_x, box_y, color='black', linewidth=3.0,
                    transform=rot_transform, zorder=8)
        ax_map.text(ux, uy + half_width_m * 1.3, f"({letter})",
                    transform=rot_transform, fontsize=16, fontweight='bold',
                    ha='center', va='bottom', zorder=9,
                    path_effects=[pe.Stroke(linewidth=2.5, foreground='white'), pe.Normal()])

        # fraction ombree dans la boite
        frac_shadow = np.nan
        if shadow_vals is not None:
            in_box = ((x_2d >= ux - half_width_m) & (x_2d <= ux + half_width_m) &
                      (y_2d >= uy - half_width_m) & (y_2d <= uy + half_width_m))
            if in_box.any():
                frac_shadow = float(np.mean(shadow_vals[in_box] > 0.5))
        shadow_fractions[letter] = frac_shadow

        # profil
        xf, yf, dist_f = trace_flowline(ux, uy)
        pts = np.column_stack([yf, xf])
        z_flow = elev_interp(pts)
        doy_flow = doy_interp(pts)

        # pente le long du profil, lissee en ignorant les NaN
        window_m = 100
        step_m = flow_length_m / n_flow_points
        window_pts = max(3, int(round(window_m / step_m)))
        ok_z = np.isfinite(z_flow).astype(float)
        num = uniform_filter1d(np.nan_to_num(z_flow), size=window_pts)
        den = uniform_filter1d(ok_z, size=window_pts)
        z_smooth = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        dz_ds = np.gradient(z_smooth, dist_f)
        slope_along_flow = np.degrees(np.arctan(np.abs(dz_ds)))

        ax_prof = profile_axes[rank]
        valid_pts = np.isfinite(z_flow) & np.isfinite(doy_flow)
        if not valid_pts.any():
            print(f"[warn] site ({letter}) {lon:.3f}E {lat:.3f}N : aucun point valide "
                  f"(hors emprise du DEM ?)")
        ax_prof.scatter(dist_f[valid_pts], z_flow[valid_pts],
                        c=doy_flow[valid_pts], cmap=cyclic_cmap,
                        vmin=1, vmax=365, s=10, edgecolor='none', zorder=3)
        ax_prof.plot(dist_f, z_flow, color='k', alpha=0.3, linewidth=0.7, zorder=2)
        ax_prof.set_xlabel("Distance (m)", fontsize=16)
        ax_prof.set_ylabel("Elevation (m)", fontsize=16)
        ax_prof.tick_params(labelsize=12)
        ax_prof.grid(linestyle='--', alpha=0.4)

        suffix = ""
        if np.isfinite(frac_shadow) and frac_shadow > 0.05:
            suffix = f"  \u2014  {frac_shadow * 100:.0f}% shadowed"
        ax_prof.set_title(f"({letter})  {lon:.3f}\u00b0E, {lat:.3f}\u00b0N{suffix}",
                          fontsize=18)

        ax_slope = ax_prof.twinx()
        ax_slope.plot(dist_f, slope_along_flow, color='green', linestyle='--',
                      linewidth=1.2, alpha=0.85, zorder=4)
        ax_slope.set_ylabel("Along-flow slope (\u00b0)", color='green', fontsize=16)
        ax_slope.tick_params(axis='y', labelcolor='green', labelsize=12)
        ax_slope.set_ylim(bottom=0)

    # ------------------------------------------------------------------
    # Colorbar commune
    # ------------------------------------------------------------------
    cbar_ax = fig.add_axes([0.52, 0.01, 0.44, 0.025])
    sm = plt.cm.ScalarMappable(cmap=cyclic_cmap, norm=plt.Normalize(vmin=1, vmax=365))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal',
                        ticks=[1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    cbar.set_ticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    cbar.set_label("Month of maximum velocity", fontsize=16)

    if shadow_vals is not None:
        print("Fraction ombree par boite :")
        for k in sorted(shadow_fractions):
            print(f"  ({k}) : {shadow_fractions[k] * 100:5.1f}%")

    save(fig, "fig1_slope_breaks_map_and_profiles")
    plt.close(fig)
    return shadow_fractions




if __name__ == "__main__":
    plot_slope_breaks_map_and_profiles(
        d.elevation, d.slope, d.max_peak_doy, d.base_mask,
        shadow_mask=None,
        min_slope1=0, max_slope1=12, min_slope2=18, max_slope2=26,
        rose_path=ROSE_PATH)
