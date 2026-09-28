"""
Figure : carte des ruptures de pente (A) + profils le long des lignes
d'ecoulement (B, C, D, ...).

Corrections par rapport a la version initiale :
  1. hillshade : NaN bouches par plus proche voisin (au lieu de 0, qui creait
     des falaises artificielles), dx/dy pris sur la vraie resolution,
     interpolation 'nearest' en espace donnees (le 'bilinear' apres rotation
     affine produisait un moire diagonal), extent sur les bords de pixels.
  2. bins de pente : imshow categoriel au lieu de contourf sur un masque
     booleen -> alignement exact au pixel, plus d'escaliers arrondis, et plus
     de cf.collections (deprecie en matplotlib >= 3.8).
  3. panel A : ajout d'une barre d'echelle (la rotation est rigide, donc les
     longueurs sont exactes) et affichage de la legende des bins.
  4. masque d'ombre : grise sur la carte + fraction ombree de chaque boite
     reportee dans le titre du panel correspondant.

Usage :
    from plot_slope_breaks import plot_slope_breaks_map_and_profiles
    plot_slope_breaks_map_and_profiles(elevation, slope, max_peak_doy,
                                       base_mask, shadow_mask=shadow_mask)
"""

from pathlib import Path
from utils import *
from data_exploration import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib import transforms
from matplotlib.colors import LightSource
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from PIL import Image
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, uniform_filter1d


# ---------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------
BASE_DIR = Path(r"C:/Users/zellerma/Documents/PhD/Recherche/seasonality_satellite_mtblanc")
FIG_DIR = BASE_DIR / "figures"
ROSE_PATH = FIG_DIR / "Compass_rose_simple.png"

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


def plot_slope_breaks_map_and_profiles(
        elevation, slope, max_peak_doy, base_mask,
        shadow_mask=None,
        min_slope1=0, max_slope1=12, min_slope2=18, max_slope2=26,
        candidates=None, half_width_m=800,
        flow_length_m=1200, n_flow_points=120,
        rotation_applied=45, scalebar_m=5000, slope_alpha=0.75,
        shadow_alpha=0.28, fig_dir=FIG_DIR, rose_path=ROSE_PATH):

    if candidates is None:
        candidates = [
            (6.982, 45.959),
            (7.033, 45.917),
            # (7.055, 45.897),
            (6.932, 45.878),
            (6.812, 45.847),
            # (6.754, 45.760),
        ]
    n_candidates = len(candidates)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

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

    fig.savefig(fig_dir / "slope_breaks_map_and_profiles.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(fig_dir / "slope_breaks_map_and_profiles.png", bbox_inches='tight', dpi=300)
    fig.savefig(fig_dir / "slope_breaks_map_and_profiles.svg", bbox_inches='tight', dpi=300)

    if shadow_vals is not None:
        print("Fraction ombree par boite :")
        for k in sorted(shadow_fractions):
            print(f"  ({k}) : {shadow_fractions[k] * 100:5.1f}%")

    print("plot_slope_breaks_map_and_profiles Done !")
    plt.close(fig)
    return shadow_fractions


"""
Diagnostic : quels masques couvrent les sites (B), (C), (D), (E) ?

Pour chaque boite d'echantillonnage, calcule la fraction de pixels RETENUS par
chaque critere de filtrage (convention True = pixel valide), rapportee au
nombre de pixels de base_mask presents dans la boite.

L'ordre et les lettres reproduisent exactement ceux de la figure : les sites
sont tries du nord au sud dans le repere tourne de 45 deg.

Usage :
    from check_masks_at_sites import check_masks_at_sites

    masks = {
        "mask (final)":  result_mask,
        "xcount":        mask_xcount,
        "shadow":        mask_shadow,
        "velavg":        mask_velavg,
        "snr":           mask_snr,
        "stable_areas":  mask_stable_areas,
    }
    df = check_masks_at_sites(base_mask, masks)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


DEFAULT_CANDIDATES = [
    (6.982, 45.959),
    (7.033, 45.917),
    # (7.055, 45.897),
    (6.932, 45.878),
    (6.812, 45.847),
    # (6.754, 45.760),
]


def check_masks_at_sites(base_mask, masks, candidates=None,
                         half_width_m=800, rotation_applied=45,
                         csv_path=None, verbose=True):
    """
    Parametres
    ----------
    base_mask : xarray.DataArray booleen (y, x), EPSG:32632.
    masks : dict {nom: DataArray booleen}, meme grille que base_mask.
    candidates : liste de (lon, lat). Doit etre la MEME que celle passee a la
        figure, sinon les lettres ne correspondront pas.
    half_width_m : demi-cote des boites (identique a la figure).
    rotation_applied : rotation de la figure, sert uniquement a retrouver
        l'ordre des lettres.

    Retour
    ------
    pandas.DataFrame indexe par lettre de site, en pourcentages.
    """
    if candidates is None:
        candidates = DEFAULT_CANDIDATES
    n = len(candidates)

    # --- orientation coherente ---------------------------------------
    def ensure_ascending_y(da):
        yv = da["y"].values
        if yv.size > 1 and yv[1] < yv[0]:
            return da.isel(y=slice(None, None, -1))
        return da

    base_mask = ensure_ascending_y(base_mask)
    masks = {k: ensure_ascending_y(v) for k, v in masks.items()}

    x_vals = base_mask.x.values
    y_vals = base_mask.y.values
    x_2d, y_2d = np.meshgrid(x_vals, y_vals)
    base = base_mask.values.astype(bool)

    # --- ordre des sites, identique a la figure ------------------------
    def rotate_xy(xs, ys, cx, cy, angle_deg):
        th = np.radians(angle_deg)
        xc, yc = np.asarray(xs) - cx, np.asarray(ys) - cy
        return (xc * np.cos(th) - yc * np.sin(th) + cx,
                xc * np.sin(th) + yc * np.cos(th) + cy)

    cx, cy = x_2d[base].mean(), y_2d[base].mean()
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    cand_utm = [to_utm.transform(lon, lat) for lon, lat in candidates]
    rot_y = [rotate_xy(np.array([ux]), np.array([uy]), cx, cy,
                       rotation_applied)[1][0] for ux, uy in cand_utm]
    order = np.argsort(rot_y)[::-1]
    letters = [chr(ord("B") + i) for i in range(n)]

    # --- couverture globale, pour verifier la convention ---------------
    global_cov = {}
    n_base = int(base.sum())
    for name, m in masks.items():
        mv = m.values.astype(bool)
        global_cov[name] = 100.0 * float((mv & base).sum()) / max(n_base, 1)

    # --- boucle sur les sites ------------------------------------------
    rows = []
    for rank, idx in enumerate(order):
        letter = letters[rank]
        lon, lat = candidates[idx]
        ux, uy = cand_utm[idx]
        hw = half_width_m

        in_box = ((x_2d >= ux - hw) & (x_2d <= ux + hw) &
                  (y_2d >= uy - hw) & (y_2d <= uy + hw))
        box_base = in_box & base
        n_box = int(box_base.sum())

        n_in_box = int(in_box.sum())
        row = {"site": letter, "lon": lon, "lat": lat,
               "n_pixels_base": n_box,
               "base_coverage_%": 100.0 * n_box / max(n_in_box, 1)}
        if n_box == 0:
            for name in masks:
                row[name] = np.nan
        else:
            for name, m in masks.items():
                mv = m.values.astype(bool)
                row[name] = 100.0 * float((mv & box_base).sum()) / n_box
        rows.append(row)

    df = pd.DataFrame(rows).set_index("site")

    if verbose:
        print("Convention : valeurs = % de pixels RETENUS par le critere")
        print("(un chiffre bas = site majoritairement ecarte par ce masque)\n")
        print("Couverture globale sur base_mask :")
        for name, v in global_cov.items():
            print(f"  {name:<16s} {v:6.1f} %")
        print()
        show = df.copy()
        show["site (lon, lat)"] = [f"({s})  {show.loc[s, 'lon']:.3f}E "
                                   f"{show.loc[s, 'lat']:.3f}N" for s in show.index]
        cols = ["site (lon, lat)", "n_pixels_base"] + list(masks.keys())
        with pd.option_context("display.width", 250,
                               "display.max_columns", None,
                               "display.float_format", "{:.1f}".format):
            print(show[cols].to_string(index=False))
        print()
        _print_summary(df, masks)

    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, float_format="%.2f")
        print(f"[ok] {csv_path}")

    return df


def _print_summary(df, masks, threshold=50.0):
    """Phrase prete a coller, listant les sites majoritairement ecartes."""
    print(f"Sites dont moins de {threshold:.0f} % des pixels sont retenus :")
    any_flag = False
    for name in masks:
        low = [s for s in df.index if np.isfinite(df.loc[s, name])
               and df.loc[s, name] < threshold]
        if low:
            any_flag = True
            frac = ", ".join(f"({s}) {df.loc[s, name]:.0f} %" for s in low)
            print(f"  {name:<16s} -> {frac}")
    if not any_flag:
        print("  aucun")


if __name__ == "__main__":
    plot_slope_breaks_map_and_profiles(
        elevation, slope, max_peak_doy, base_mask,
        shadow_mask=None,
        min_slope1=0, max_slope1=12, min_slope2=18, max_slope2=26,
        candidates=None, half_width_m=800,
        flow_length_m=1200, n_flow_points=120,
        rotation_applied=45, scalebar_m=5000, slope_alpha=0.75,
        shadow_alpha=0.28, fig_dir=FIG_DIR, rose_path=ROSE_PATH)


    masks = {
        "mask (final)":  result_mask,
        "xcount":        mask_xcount,
        "shadow":        mask_shadow,
        "velavg":        mask_velavg,
        "snr":           mask_snr,
        "stable_areas":  mask_stable_areas,
    }

    df = check_masks_at_sites(base_mask, masks, csv_path=FIG_DIR / "mask_coverage_sites.csv")
 