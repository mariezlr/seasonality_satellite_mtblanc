#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conceptual model figure.

    (A) (B)  friction laws            -> plot_friction_laws()
    (C)      distributed drainage     -> draw_network_panel()

Panel (C) is generated, not drawn by hand. A rough bed is synthesised, ice
slides towards +x, cavities are the shadows cast in the lee of the bumps,
cavities close enough to each other form a mechanically linked cluster, and
conduits follow the least-cost path between neighbouring clusters, crossing
the bumps only at their lowest cols.

Reproduce with:      python conceptual_model_figure.py
Dependencies:        numpy, scipy, matplotlib
"""
from __future__ import annotations

from utils import *
from data_exploration import *

import heapq
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow, Polygon, Rectangle
from scipy.ndimage import (center_of_mass, distance_transform_edt,
                           gaussian_filter, label, uniform_filter)
import matplotlib.ticker as mticker
from scipy.ndimage import binary_fill_holes



FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_STEM = "CN_conceptual_model"

plt.rcParams["contour.negative_linestyle"] = "solid"

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



# ======================================================================
# 1. PARAMETERS
# ======================================================================
@dataclass
class Config:
    """Every knob of panel (C). Lengths in metres."""

    seed: int = 2
    lx: float = 120.0                 # domain, x is the ice-flow direction
    ly: float = 40.0
    nx: int = 640
    ny: int = int(nx * ly / lx)

    # --- bed: self-affine, band-limited
    hurst: float = 0.75
    z_rms: float = 0.9                # standard deviation of bed elevation
    lam_min: float = 2.2              # fine cut-off: larger -> smoother cavities
    lam_max: float = 35.0             # coarse cut-off: sets the spacing of lows

    # --- cavities: shadowing of the sliding ice
    s_sep: float = 0.40               # reattachment slope: smaller -> more cavities
    s_hollow: float = 0.50            # separation favoured in lows (0 = uniform)
    d_min: float = 0.02               # wet threshold
    smooth_cav: float = 0.45          # smoothing of the cavity outline
    a_min: float = 5.0                # smallest cavity drawn (m2)
    a_cluster: float = 15.0           # smallest cluster (m2)
    group: float = 1                # reach of the mechanical link between cavities

    # --- conduits
    link_max: float = 65.0            # longest link between two clusters
    conduit_w: float = 1.2            # conduit half-width
    cost_h: float = 0.55              # elevation cost stiffness: small -> sinuous
    outlets: bool = True              # inlet and outlet at the domain edges
    inside_reach: float = 0.6         # conduits fade out this far inside a cluster
    inside_strength: float = 1.0
    a_con_min: float = 8.0            # smallest conduit stretch coloured green

    # --- idealised cavities ("potatoes")
    pot_len: tuple = (4.0, 9.0)       # length along flow
    pot_aspect: tuple = (1.5, 2.3)    # length / width
    pot_tilt: float = 12.0            # scatter around the flow direction (deg)
    pot_wobble: float = 0.12          # outline irregularity (0 = ellipse)
    pot_gap: float = 0.25             # smallest gap between two potatoes
    pot_tries: int = 40000
    pot_fit: float = 0.25             # fraction of the length that must fit inside

    # --- rendering
    halo: float = 1.6                 # thickness of the cluster envelope
    lvl_halo: float = 0.30            # contour level of the envelope
    bed_smooth: float = 2.4           # background smoothing (rendering only)
    n_contours: int = 11
    scale_bar: float = 20.0
    ann_fs: int = 14                  # annotation font size
    arrow_gap: float = 1.5            # gap between label and arrow tail
    off_x: float = 5.0                # shifts the window towards the right (m)
    off_y: float = 4.5                # shifts towards the top (m)

    # --- manual overrides of the annotations (None = automatic), in metres
    con_text_xy: tuple | None = None
    con_targets: list | None = None
    con_min_sep: float = 10.0         # two labelled conduits at least this far apart
    clu_text_xy: tuple | None = None
    clu_targets: list | None = None
    n_clu_arrows: int = 3
    a_hole_min: float = 12.0     # dans Config : trou torique conserve au-dela (m2)

    # --- colours
    c_bed: tuple = ("#FAFAF8", "#EDEDEA", "#DEDEDA", "#CBCBC6", "#B6B6B0")
    c_cav: str = "#3D7FA8"
    c_cav_edge: str = "#1C4A63"
    c_clu_light: str = "#AFD4E6"
    c_clu_dark: str = "#2E7BA6"
    c_con_light: str = "#F5C87A"
    c_con_dark: str = "#C77A1E"
    c_con_edge: str = "#8F5410"     # contour des conduits, plus fonce
    c_edge: str = "#1C4A63"         # contour des grappes, bleu fonce
    c_contour: str = "#9A9A93"

    c_winter_6: str = "#E8A94A"    # ambre clair  : pente faible
    c_winter_24: str = "#8F5410"   # ambre fonce  : pente forte

    # --- derived
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.x = np.linspace(0, self.lx, self.nx)
        self.y = np.linspace(0, self.ly, self.ny)

    @property
    def dx(self):
        return self.x[1] - self.x[0]

    @property
    def dy(self):
        return self.y[1] - self.y[0]

    @property
    def grid(self):
        return np.meshgrid(self.x, self.y)


# ======================================================================
# 2. GEOMETRY
# ======================================================================
def make_bed(cfg: Config, rng) -> np.ndarray:
    """Band-limited self-affine bed."""
    kx = np.fft.fftfreq(cfg.nx, d=cfg.lx / cfg.nx)
    ky = np.fft.fftfreq(cfg.ny, d=cfg.ly / cfg.ny)
    k = np.hypot(*np.meshgrid(kx, ky))
    k[0, 0] = 1e-9
    amp = k ** (-(cfg.hurst + 1.0))
    amp[(k > 1.0 / cfg.lam_min) | (k < 1.0 / cfg.lam_max)] = 0.0
    z = np.real(np.fft.ifft2(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, k.shape))))

    z = np.roll(z, (-int(round(cfg.off_y / cfg.dy)), 
                    -int(round(cfg.off_x / cfg.dx))), axis=(0, 1))
    
    return (z - z.mean()) * (cfg.z_rms / z.std())


def make_cavities(cfg: Config, z: np.ndarray):
    """Shadow cast by the sliding ice; separation favoured in large-scale lows."""
    zbig = uniform_filter(z, size=(int(12 / cfg.dy), int(16 / cfg.dx)),
                          mode="nearest")
    sfac = np.exp(cfg.s_hollow * (zbig - zbig.mean()) / zbig.std())

    sole = z.copy()
    for i in range(1, cfg.nx):
        sole[:, i] = np.maximum(z[:, i],
                                sole[:, i - 1] - cfg.s_sep * sfac[:, i] * cfg.dx)
    d_cav = gaussian_filter(np.maximum(sole - z, 0.0), cfg.smooth_cav / cfg.dx)

    lb, _ = label(d_cav > cfg.d_min)
    area = np.bincount(lb.ravel()) * cfg.dx * cfg.dy
    return np.where((lb > 0) & (area[lb] > cfg.a_min), d_cav, 0.0)


def make_clusters(cfg: Config, d_cav: np.ndarray):
    """Cavities closer than `group` are mechanically linked: they form a cluster."""
    wet = d_cav > cfg.d_min
    grp = gaussian_filter(wet.astype(float), cfg.group / cfg.dx) > 0.20
    lbz, nz = label(grp)
    keep = [k for k in range(1, nz + 1)
            if (wet & (lbz == k)).sum() * cfg.dx * cfg.dy > cfg.a_cluster]
    cluster = np.isin(lbz, keep)

    centres = sorted(((cfg.x[int(i)], cfg.y[int(j)])
                      for j, i in (center_of_mass(lbz == k) for k in keep)),
                     key=lambda p: p[0])
    return cluster, centres, lbz, keep


def _least_cost_path(cfg: Config, z: np.ndarray, p0, p1, step=3):
    """Path minimising the integrated elevation cost: goes round bumps."""
    zc, xc, yc = z[::step, ::step], cfg.x[::step], cfg.y[::step]
    nyc, nxc = zc.shape
    w = np.exp((zc - zc.min()) / cfg.cost_h)

    j0, i0 = int(np.argmin(abs(yc - p0[1]))), int(np.argmin(abs(xc - p0[0])))
    j1, i1 = int(np.argmin(abs(yc - p1[1]))), int(np.argmin(abs(xc - p1[0])))
    best = np.full((nyc, nxc), np.inf)
    prev = np.full((nyc, nxc), -1, dtype=int)
    best[j0, i0] = 0.0
    heap = [(0.0, j0 * nxc + i0)]
    while heap:
        c, node = heapq.heappop(heap)
        j, i = divmod(node, nxc)
        if c > best[j, i]:
            continue
        if (j, i) == (j1, i1):
            break
        for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0),
                       (1, 1), (-1, 1), (1, -1), (-1, -1)):
            jj, ii = j + dj, i + di
            if 0 <= jj < nyc and 0 <= ii < nxc:
                length = np.hypot(dj * cfg.dy * step, di * cfg.dx * step)
                nc = c + 0.5 * (w[j, i] + w[jj, ii]) * length
                if nc < best[jj, ii]:
                    best[jj, ii] = nc
                    prev[jj, ii] = node
                    heapq.heappush(heap, (nc, jj * nxc + ii))

    pts, node = [], j1 * nxc + i1
    while node != -1:
        j, i = divmod(node, nxc)
        pts.append((xc[i], yc[j]))
        node = prev[j, i]
    pts = np.array(pts[::-1])
    if len(pts) > 14:                                   # smooth the staircase
        ker = np.ones(9) / 9.0
        pts = np.column_stack([np.convolve(pts[:, 0], ker, "valid"),
                               np.convolve(pts[:, 1], ker, "valid")])
        pts = np.vstack([p0, pts, p1])
    return pts


def _tube(cfg: Config, xg, yg, pts) -> np.ndarray:
    """Distance to a polyline."""
    d = np.full(xg.shape, 1e9)
    for a, b in zip(pts[:-1], pts[1:]):
        vx, vy = b[0] - a[0], b[1] - a[1]
        l2 = vx * vx + vy * vy
        if l2 < 1e-9:
            continue
        t = np.clip(((xg - a[0]) * vx + (yg - a[1]) * vy) / l2, 0, 1)
        d = np.minimum(d, np.hypot(xg - (a[0] + t * vx), yg - (a[1] + t * vy)))
    return d


def make_conduits(cfg: Config, z, xg, yg, cluster, centres):
    """One conduit per pair of neighbouring clusters, plus inlet and outlet."""
    links = [(centres[a], centres[b])
             for a in range(len(centres)) for b in range(a + 1, len(centres))
             if np.hypot(centres[a][0] - centres[b][0],
                         centres[a][1] - centres[b][1]) < cfg.link_max]
    tracks = [_least_cost_path(cfg, z, a, b) for a, b in links]
    n_link = len(tracks)
    if cfg.outlets and centres:
        tracks.append(np.array([[-0.05 * cfg.lx, centres[0][1]], list(centres[0])]))
        tracks.append(np.array([list(centres[-1]), [1.05 * cfg.lx, centres[-1][1]]]))

    f_con = np.zeros_like(xg)
    for pts in tracks:
        f_con += np.exp(-(_tube(cfg, xg, yg, pts) / cfg.conduit_w) ** 2)

    # a conduit has no separate existence inside a cluster: fade it out
    inside = gaussian_filter(cluster.astype(float), cfg.inside_reach / cfg.dx)
    f_con *= np.clip(1.0 - cfg.inside_strength * inside, 0, 1)
    return tracks, n_link, f_con


def water_fields(cfg, cluster, f_con, lbz, keep):
    """Envelope of the water sheet, and the purple-to-green mixing index."""
    f_clu = np.zeros_like(f_con)
    for k in keep:                       # une enveloppe par grappe, sans cumul
        f_clu = np.maximum(
            f_clu, gaussian_filter((lbz == k).astype(float), cfg.halo / cfg.dx) * 2.2)
    f_halo = f_clu + f_con

    wet = f_halo > cfg.lvl_halo
    holes = binary_fill_holes(wet) & ~wet
    lbh, _ = label(holes)
    ah = np.bincount(lbh.ravel()) * cfg.dx * cfg.dy
    small = (lbh > 0) & (ah[lbh] < cfg.a_hole_min)
    f_halo = np.where(small, cfg.lvl_halo + 1e-3, f_halo)
    f_halo = gaussian_filter(f_halo, 0.4 / cfg.dx)

    # green only where the water would not exist without the conduit
    t = np.clip((cfg.lvl_halo + 0.02 - f_clu) / 0.10, 0, 1)
    mixt = t * np.clip(f_con / 0.35, 0, 1)

    # drop the green slivers left on cluster rims
    green = (f_halo > cfg.lvl_halo) & (mixt > 0.5)
    lbc, _ = label(green)
    ac = np.bincount(lbc.ravel()) * cfg.dx * cfg.dy
    big = (lbc > 0) & (ac[lbc] >= cfg.a_con_min)
    keep = np.clip(3.0 * gaussian_filter(big.astype(float), 1.0 / cfg.dx), 0, 1)
    return f_halo, mixt * keep


def make_potatoes(cfg: Config, xg, yg, d_cav, cluster, f_con, rng):
    """Idealised cavities: smooth blobs elongated along the ice flow."""
    mask = cluster
    con_zone = f_con > 0.08                       # drawn conduit, plus a margin
    jj, ii = np.nonzero(mask)
    dref = np.percentile(d_cav[d_cav > cfg.d_min], 90)

    def outline(cx, cy, length, width, tilt):
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        r = np.ones_like(t)
        for k in (2, 3, 5):
            r += cfg.pot_wobble / k * np.cos(k * t + rng.uniform(0, 2 * np.pi))
        px, py = 0.5 * length * r * np.cos(t), 0.5 * width * r * np.sin(t)
        c, s = np.cos(tilt), np.sin(tilt)
        return np.column_stack([cx + c * px - s * py, cy + s * px + c * py])

    def inside(px, py):
        i, j = int(round(px / cfg.dx)), int(round(py / cfg.dy))
        return 0 <= i < cfg.nx and 0 <= j < cfg.ny and mask[j, i]

    def clear_of_conduit(pg):
        i = np.clip(np.round(pg[:, 0] / cfg.dx).astype(int), 0, cfg.nx - 1)
        j = np.clip(np.round(pg[:, 1] / cfg.dy).astype(int), 0, cfg.ny - 1)
        return not con_zone[j, i].any()

    placed = np.zeros((0, 4))                     # cx, cy, length, width
    polys = []
    for _ in range(cfg.pot_tries):
        k = rng.integers(len(jj))
        cx = cfg.x[ii[k]] + rng.uniform(-0.5, 0.5) * cfg.dx
        cy = cfg.y[jj[k]] + rng.uniform(-0.5, 0.5) * cfg.dy
        f = np.clip(d_cav[jj[k], ii[k]] / dref, 0, 1) ** 0.5     # deep -> large
        length = cfg.pot_len[0] + (cfg.pot_len[1] - cfg.pot_len[0]) * f \
            * rng.uniform(0.75, 1.0)
        width = length / rng.uniform(*cfg.pot_aspect)
        if not (inside(cx - cfg.pot_fit * length, cy)
                and inside(cx + cfg.pot_fit * length, cy)):
            continue
        if len(placed) and np.any(
                ((cx - placed[:, 0]) / (0.5 * (length + placed[:, 2]) + cfg.pot_gap)) ** 2
                + ((cy - placed[:, 1]) / (0.5 * (width + placed[:, 3]) + cfg.pot_gap)) ** 2
                <= 1.0):
            continue
        tilt = np.deg2rad(rng.uniform(-cfg.pot_tilt, cfg.pot_tilt))
        pg = outline(cx, cy, length, width, tilt)
        if not clear_of_conduit(np.vstack([pg, [[cx, cy]]])):
            continue
        placed = np.vstack([placed, [cx, cy, length, width]])
        polys.append(pg)
    return polys


# ======================================================================
# 3. ANNOTATIONS OF PANEL (C)
# ======================================================================
def _conduit_targets(cfg: Config, tracks, n_link, cluster, f_con):
    """Middle of each visible conduit stretch, two of them, far enough apart."""
    def cell(px, py):
        return (int(np.clip(round(py / cfg.dy), 0, cfg.ny - 1)),
                int(np.clip(round(px / cfg.dx), 0, cfg.nx - 1)))

    cand = []
    for pts in tracks[:n_link]:
        ok = [(not cluster[cell(*p)]) and f_con[cell(*p)] > 0.5 for p in pts]
        k = 0
        while k < len(pts):
            if ok[k]:
                e = k
                while e + 1 < len(pts) and ok[e + 1]:
                    e += 1
                if e - k >= 3:
                    cand.append(tuple(pts[(k + e) // 2]))
                k = e + 1
            else:
                k += 1

    chosen = []
    for p in sorted(cand, key=lambda p: -p[1]):          # topmost first
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= cfg.con_min_sep
               for q in chosen):
            chosen.append(p)
        if len(chosen) == 2:
            break
    return chosen


def _cluster_anchor(cfg: Config, xg, yg, f_halo, lbz, keep):
    """Driest spot near the centre, and the nearest rim of each close cluster."""
    dist = distance_transform_edt(~(f_halo > cfg.lvl_halo)) * cfg.dx
    wc = np.exp(-(((xg - cfg.lx / 2) / (0.3 * cfg.lx)) ** 2
                  + ((yg - cfg.ly / 2) / (0.3 * cfg.ly)) ** 2))
    j, i = np.unravel_index(np.argmax(dist * wc), dist.shape)
    txy = (cfg.x[i], cfg.y[j])

    cand = []
    for k in keep:
        jj, ii = np.nonzero(lbz == k)
        d = np.hypot(cfg.x[ii] - txy[0], cfg.y[jj] - txy[1])
        m = int(np.argmin(d))
        cand.append((d[m], cfg.x[ii[m]], cfg.y[jj[m]]))
    targets = [(px, py) for _, px, py in sorted(cand)[:cfg.n_clu_arrows]]
    return txy, targets


def _annotate(cfg: Config, ax, txy, txt, targets, colour):
    """Label plus radiating arrows that start clear of the text."""
    lab = ax.text(*txy, txt, ha="center", va="center", multialignment="center",
                  linespacing=1.15, fontsize=cfg.ann_fs, color=colour,
                  fontweight="bold", zorder=12)
    ax.figure.canvas.draw()                      # freeze the text extent
    bb = lab.get_window_extent(
        renderer=ax.figure.canvas.get_renderer()).transformed(
        ax.transData.inverted())
    cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
    hw, hh = (bb.x1 - bb.x0) / 2, (bb.y1 - bb.y0) / 2
    for px, py in targets:
        ux, uy = px - cx, py - cy
        norm = np.hypot(ux, uy)
        ux, uy = ux / norm, uy / norm
        s = min(hw / abs(ux) if abs(ux) > 1e-9 else np.inf,
                hh / abs(uy) if abs(uy) > 1e-9 else np.inf)
        start = (cx + (s + cfg.arrow_gap) * ux, cy + (s + cfg.arrow_gap) * uy)
        ax.annotate("", xy=(px, py), xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.8,
                                    mutation_scale=16, shrinkA=0, shrinkB=1,
                                    connectionstyle="arc3,rad=0.12"),
                    annotation_clip=False, zorder=11)


# ======================================================================
# 4. PANEL (C)
# ======================================================================
def draw_network_panel(ax, cfg: Config | None = None, verbose: bool = True):
    cfg = cfg or Config()
    rng = np.random.default_rng(cfg.seed)
    xg, yg = cfg.grid

    z = make_bed(cfg, rng)
    d_cav = make_cavities(cfg, z)
    cluster, centres, lbz, keep = make_clusters(cfg, d_cav)
    tracks, n_link, f_con = make_conduits(cfg, z, xg, yg, cluster, centres)
    f_halo, mixt = water_fields(cfg, cluster, f_con, lbz, keep)
    potatoes = make_potatoes(cfg, xg, yg, d_cav, cluster, f_con, rng)

    # --- bed: banded relief plus contour lines
    cmap_bed = LinearSegmentedColormap.from_list("bed", list(cfg.c_bed))
    zs = gaussian_filter(z, cfg.bed_smooth / cfg.dx)     # rendering only
    ax.contourf(xg, yg, zs, levels=cfg.n_contours, cmap=cmap_bed, zorder=0)
    ax.contour(xg, yg, zs, levels=cfg.n_contours, colors=cfg.c_contour,
               linewidths=0.7, alpha=0.65, zorder=1)

    # --- water sheet, purple (cluster) to orange (conduit)
    cmap_w = LinearSegmentedColormap.from_list(
        "water", [cfg.c_clu_light, "#B0CCDC", "#9FC7B8", cfg.c_con_light])
    sheet = np.where(f_halo > cfg.lvl_halo, np.clip(mixt, 0, 1), np.nan)
    ax.contourf(xg, yg, sheet, levels=np.linspace(0, 1, 15), cmap=cmap_w,
                vmin=0, vmax=1, zorder=2, extend="both", alpha=0.5)
    
    # edge = ax.contour(xg, yg, f_halo, levels=[cfg.lvl_halo],
    #                   colors=[cfg.c_edge], linewidths=1.4, zorder=3)
    # edge.set(path_effects=[pe.withStroke(linewidth=6, foreground="white",
    #                                      alpha=0.55)])

    # bord des grappes
    mask_clu = np.where(mixt <= 1.0, f_halo, np.nan)
    e1 = ax.contour(xg, yg, mask_clu, levels=[cfg.lvl_halo],
                    colors=[cfg.c_edge], linewidths=1.4, zorder=3)
    # bord des conduits
    mask_con = np.where(mixt > 0.01, f_halo, np.nan)
    e2 = ax.contour(xg, yg, mask_con, levels=[cfg.lvl_halo],
                    colors=[cfg.c_con_edge], linewidths=1.4, zorder=3)
    for e in (e1, e2):
        e.set(path_effects=[pe.withStroke(linewidth=6, foreground="white",
                                          alpha=0.55)])
    # ax.contour(xg, yg, f_halo, levels=[cfg.lvl_halo], colors=[cfg.c_edge],
    #            linewidths=1.4, zorder=3)
    # edge_band = np.abs(f_halo - cfg.lvl_halo) < 0.06        # proche du seuil
    # mask_con = (mixt > 0.25) & edge_band
    # ax.contour(xg, yg, np.where(mask_con, f_halo, cfg.lvl_halo - 1.0),
    #            levels=[cfg.lvl_halo], colors=[cfg.c_con_edge],
    #            linewidths=1.4, zorder=4)
        

    # --- idealised cavities
    for pg in potatoes:
        ax.add_patch(Polygon(pg, closed=True, facecolor=cfg.c_cav,
                             edgecolor=cfg.c_cav_edge, lw=0.8, alpha=0.6,
                             zorder=4))

    # --- ice flow
    ax.add_patch(FancyArrow(0.035 * cfg.lx, 0.920 * cfg.ly, 0.12 * cfg.lx, 0,
                            width=0.6, head_width=2.3, head_length=2.8,
                            color="#000000", length_includes_head=True,
                            zorder=12))
    ax.text(0.036 * cfg.lx, 0.945 * cfg.ly, "Ice flow", fontsize=16,
            color="#000000", fontweight="bold", zorder=12)

    # --- scale bar
    bx, by = 0.035 * cfg.lx, 0.040 * cfg.ly
    ax.add_patch(Rectangle((bx - 2, by - 1), cfg.scale_bar + 4, 0.10 * cfg.ly,
                           facecolor="white", alpha=0.85, edgecolor="none",
                           zorder=9))
    ax.add_patch(Rectangle((bx, by), cfg.scale_bar, 0.016 * cfg.ly,
                           facecolor="#2C3238", zorder=10))
    ax.text(bx + cfg.scale_bar / 2, by + 0.028 * cfg.ly,
            "%d m" % cfg.scale_bar, ha="center", fontsize=14,
            fontweight="bold", color="#2C3238", zorder=10)

    ax.set_xlim(0, cfg.lx)
    ax.set_ylim(0, cfg.ly)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- labels with arrows (after the axes are final: they use screen extents)
    con_targets = cfg.con_targets or _conduit_targets(cfg, tracks, n_link,
                                                      cluster, f_con)
    con_xy = cfg.con_text_xy or (
        float(np.clip(np.mean([p[0] for p in con_targets]),
                      0.28 * cfg.lx, 0.75 * cfg.lx)), 0.93 * cfg.ly)
    clu_xy, clu_targets = _cluster_anchor(cfg, xg, yg, f_halo, lbz, keep)
    clu_xy = cfg.clu_text_xy or clu_xy
    clu_targets = cfg.clu_targets or clu_targets

    _annotate(cfg, ax, con_xy, "Conduits in\nbedrock constrictions",
              con_targets, cfg.c_con_dark)
    _annotate(cfg, ax, clu_xy, "Cavity clusters\nin bedrock lows",
              clu_targets, cfg.c_cav_edge)

    if verbose:
        print("panel C: %.0f %% wetted | %d clusters | %d conduits | %d cavities"
              % (100 * (d_cav > cfg.d_min).mean(), len(centres), n_link,
                 len(potatoes)))
        print("         conduit label", np.round(con_xy, 1),
              "targets", np.round(con_targets, 1).tolist())
        print("         cluster label", np.round(clu_xy, 1),
              "targets", np.round(clu_targets, 1).tolist())


# ======================================================================
# 5. PANELS (A) AND (B)
# ======================================================================
def plot_friction_laws(ax1, ax2, cfg: Config | None = None):
    """Friction laws. Draws on `ax1` and `ax2`, which the caller provides."""

    cfg = cfg or Config()

    print(slope_min_intersect, slope_max_intersect)
    # ax.plot(slope_line, tau_emp, linestyle='--', color='green', label=r'Average $\tau_b$')#, label=r'Average $\tau_b$ from Elmer/Ice simulations')

    ax1.plot(slope_line, CN_min, linestyle='-', color=cfg.c_clu_dark)
    ax1.plot(slope_line, CN_max, linestyle='-', color=cfg.c_clu_dark,
             label=r"$CN_{cavities}$ = $\tau_b(1/\theta)^{1/3}$ (in summer); $\theta \in $[0.4;0.6]")
    ax1.fill_between(slope_line, CN_min, CN_max,
        color=cfg.c_clu_light, alpha=0.5)

    ax1.plot(slope_line, CN_channels_min, linestyle='-', color=cfg.c_con_dark)
    ax1.plot(slope_line, CN_channels_max, linestyle='-', color=cfg.c_con_dark,
             label=fr'$CN_{{conduits}}$ = f tan$(\alpha)^{{0.47}}$ (in winter); f$ \in $[{f_min:.2f};{f_max:.2f}]')
    ax1.fill_between(slope_line, CN_channels_min, CN_channels_max,
        color=cfg.c_con_light, alpha=0.5)
    

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

    ax1.set_xlabel("Surface slope (°)")
    ax1.set_ylabel(fr"CN (MPa)")

    ax1.legend(loc = "upper left")
    ax1.grid(linestyle="--")

    sliding_vel = np.arange(1, 500, 0.1)
    As_winter = 7500
    As_summer = As_winter /(1-0.5)

    CN_6 = 0.29 * np.tan(np.radians(6))**0.47
    CN_24 = 0.29 * np.tan(np.radians(24))**0.47

    # highlight the representative tau_b ~ 0.1 MPa range
    ax2.axhspan(0.085, 0.115, alpha=0.12, color='black', zorder=1)
    ax2.axhline(y=0.1, color='black', linestyle=':', linewidth=1.2, alpha=0.5, zorder=2)

    ax2.hlines(CN_6, np.min(sliding_vel), np.max(sliding_vel), color=cfg.c_winter_6, linewidth=0.6, linestyle="--")
    ax2.text(1.1, 0.98 * CN_6, r"$CN(6^\circ)$", va="top", ha="left", fontsize=9, color=cfg.c_winter_6)
    ax2.hlines(CN_24, np.min(sliding_vel), np.max(sliding_vel), color=cfg.c_winter_24, linewidth=0.6, linestyle="--")
    ax2.text(1.1, 0.98 * CN_24, r"$CN(24^\circ)$", va="top", ha="left", fontsize=9, color=cfg.c_winter_24)

    taub_summer = [power_law(u, As_summer) for u in sliding_vel]
    taub_winter_6 = [cavitation_law(u, CN_6, 1, As_winter) for u in sliding_vel]
    taub_winter_24 = [cavitation_law(u, CN_24, 1, As_winter) for u in sliding_vel]


    ax2.plot(sliding_vel, taub_summer, color=cfg.c_clu_dark, linewidth=6, alpha = 0.9, label="Summer")
    ax2.plot(sliding_vel, taub_winter_6, color=cfg.c_winter_6, linewidth=6, alpha = 0.9, label="Winter 6°")
    ax2.plot(sliding_vel, taub_winter_24, color=cfg.c_winter_24, linewidth=6, alpha = 0.9, label="Winter 24°")


    basal_shear_stress = np.arange(0.04, 0.25, 10**(-8))

    inv_summer = interp1d(np.array(taub_summer), sliding_vel, bounds_error=False, fill_value=np.nan)
    inv_winter_6 = interp1d(np.array(taub_winter_6), sliding_vel, bounds_error=False, fill_value=np.nan)
    inv_winter_24 = interp1d(np.array(taub_winter_24), sliding_vel, bounds_error=False, fill_value=np.nan)

    ub_mean_6 = (1/3) * inv_summer(basal_shear_stress) + (2/3) * inv_winter_6(basal_shear_stress)
    ub_mean_24 = (1/3) * inv_summer(basal_shear_stress) + (2/3) * inv_winter_24(basal_shear_stress)
    
    ax2.plot(ub_mean_6, basal_shear_stress, color=cfg.c_winter_6, linestyle="--", label="Annual mean")
    ax2.plot(ub_mean_24, basal_shear_stress, color=cfg.c_winter_24, linestyle="--")

    tau_arrow = 0.09
    u_summer = inv_summer(tau_arrow)
    u_winter = inv_winter_6(tau_arrow)
    ax2.annotate("", xy=(u_summer, tau_arrow), xytext=(u_winter, tau_arrow),
        arrowprops=dict(arrowstyle="->", linewidth=2, color=cfg.c_winter_6), fontsize=12)

    ax2.text(u_winter*0.9, tau_arrow*0.86, "Winter faster", fontweight='bold', rotation=15, ha="center", va="bottom", fontsize=12, color=cfg.c_winter_6)

    tau_arrow = 0.11
    u_summer = inv_summer(tau_arrow)
    u_winter = inv_winter_24(tau_arrow)
    ax2.annotate("", xy=(u_summer, tau_arrow), xytext=(u_winter, tau_arrow),
        arrowprops=dict(arrowstyle="->", linewidth=2, color=cfg.c_winter_24), fontsize=12)

    ax2.text(u_winter*0.9, tau_arrow*0.92, "Winter slower", fontweight='bold', rotation=45, ha="center", va="bottom", fontsize=12, color=cfg.c_winter_24)

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


# ======================================================================
# 6. ASSEMBLY
# ======================================================================
def build_figure(cfg: Config | None = None, with_friction: bool = True):
    cfg = cfg or Config()
    w_fig, h_top, margins = 14.0, 6.2, 1.1
    h_bot = (w_fig - margins) * cfg.ly / cfg.lx

    if with_friction:
        fig = plt.figure(figsize=(w_fig, h_top + h_bot))
        gs = fig.add_gridspec(2, 1, height_ratios=[h_top, h_bot], hspace=0.22,
                              left=0.06, right=0.98, top=0.95, bottom=0.04)
        top = gs[0].subgridspec(1, 2, wspace=0.24)
        ax1, ax2 = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        plot_friction_laws(ax1, ax2)
        panels = ((ax1, "(A)"), (ax2, "(B)"), (ax3, "(C)"))
    else:
        fig = plt.figure(figsize=(w_fig - margins, h_bot))
        ax3 = fig.add_axes([0, 0, 1, 1])
        panels = ()

    draw_network_panel(ax3, cfg)
    for ax, lab in panels:
        ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=20,
                fontweight="bold", va="bottom", ha="left")
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fig = build_figure()
        stem = FIG_STEM
    except NotImplementedError as err:              # panels A and B not wired yet
        print("warning: %s -- panel (C) only" % err)
        fig = build_figure(with_friction=False)
        stem = "network_plan"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=200, bbox_inches="tight",
                    pad_inches=0.02)
    print("written:", FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()