"""
Shared configuration for the manuscript figure scripts.

Centralises paths and dataset loading so that `plot_fig1.py`, `plot_fig2.py`,
`plot_fig3.py` and `plot_sup.py` all start from the same state.

Paths are derived from the location of this file, so the repository can be
cloned anywhere without editing anything.

Usage
-----
    from plots_config import FIG_DIR, load_dataset

    d = load_dataset()          # only if the figure actually needs the data
    d.slope, d.vel_cycle, ...
"""

from pathlib import Path
from types import SimpleNamespace

import xarray as xr
import numpy as np

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "output"
DATASET_PATH = OUT_DIR / "analysis_dataset.nc"

FIG_DIR = PROJECT_ROOT / "figures_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ROSE_PATH = PROJECT_ROOT / "figures_paper" / "Compass_rose_simple.png"

# Summer window used to average melt rates (mid-June to mid-September)
SUMMER_DOY = (166, 258)

# Slope classes used throughout the paper (degrees)
GENTLE = (0, 12)
STEEP = (18, 36)

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
_CACHE = None


def load_dataset(path=None):
    """Open `analysis_dataset.nc` and return its variables as a namespace.

    The dataset is opened once per session and cached, so several figure
    scripts can be run from the same interpreter without reloading it.

    Returns
    -------
    SimpleNamespace with attributes:
        ds                  the raw xarray Dataset
        velocity, vel_detrended, vel_lowpass, vel_cycle, avg_velocity
        amplitude, amplitude_rel
        melt_cycle, avg_melt_summer
        slope, elevation
        max_peak_doy, min_peak_doy, inflex_doy
        base_mask, result_mask, mask_xcount, mask_shadow, mask_velavg,
        mask_snr, mask_stable_areas
    """
    global _CACHE
    if _CACHE is not None and path is None:
        return _CACHE

    path = Path(path) if path is not None else DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python processing.py` first "
            "(see README, section 'Reproducing the analysis')."
        )

    ds = xr.open_dataset(path)
    base_mask = ds["base_mask"]
    yy, xx = np.meshgrid(ds["y"].values, ds["x"].values, indexing="ij")

    # Mean summer melt rate, averaged over all years
    melt_cycle = ds["melt_cycle"]
    lo, hi = SUMMER_DOY
    melt_summer = melt_cycle.where(
        (melt_cycle["doy_approx"] > lo) & (melt_cycle["doy_approx"] <= hi)
    )

    d = SimpleNamespace(
        x_1d=xx.flatten(),
        y_1d=yy.flatten(),
        ds=ds,
        velocity=ds["velocity"],
        vel_detrended=ds["vel_detrended"],
        vel_lowpass=ds["vel_lowpass"],
        vel_cycle=ds["vel_cycle"],
        avg_velocity=ds["avg_velocity"],
        amplitude=ds["amplitude"],
        amplitude_rel=ds["amplitude_rel"],
        melt_cycle=melt_cycle,
        avg_melt_summer=melt_summer.mean(dim=["cycle"], skipna=True),
        slope=ds["slope"],
        elevation=ds["elevation"],
        max_peak_doy=ds["max_peak_doy"],
        min_peak_doy=ds["min_peak_doy"],
        inflex_doy=ds["inflex_doy"],
        base_mask=base_mask,
        # Quality masks. `result_mask` is the one used for the main results;
        # the individual masks are kept for the supplementary mask figure.
        result_mask=ds["mask"] & base_mask,
        mask_xcount=ds["mask_xcount"] & base_mask,
        mask_shadow=ds["mask_shadow"] & base_mask,
        mask_velavg=ds["mask_velavg"] & base_mask,
        mask_snr=ds["mask_snr"] & base_mask,
        mask_stable_areas=ds["mask_stable_areas"],
    )

    if path == DATASET_PATH:
        _CACHE = d
    return d


def save(fig, stem, formats=("pdf", "png"), dpi=300):
    """Write `fig` to FIG_DIR as `stem.<ext>` for each requested format."""
    for ext in formats:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)
    print(f"  -> {stem} ({', '.join(formats)})")