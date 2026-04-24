#!/usr/bin/env python3
"""
LMR_surfer.py
=============
Lower Mississippi River hydraulic channel profiler.

Creates a dual-panel diagnostic figure:

  Panel 1 — Water-Surface Elevation Profile
      • DEM-derived channel thalweg (bed) elevation (NAVD88 ft)
      • Interpolated continuous water-surface profile
      • Monitored gage markers: station ID and current corrected stage
      • Shaded water volume between bed and surface

  Panel 2 — In-Channel Flow Storage + Hysteresis
      • Stage vs. river mile at each gage
      • Historical P05–P95 stage envelope for the current discharge bin
        (shaded vertical band at each station)
      • Current date stage as a coloured marker (▲ rising / ▼ falling / ● normal)
      • Cumulative in-channel storage (acre-ft) from DEM cross-sections
        shown on a secondary right-hand axis
      • Hysteresis context: stage above historical median → rising limb (storing
        more water); stage below → falling limb (losing stored water)

Usage
-----
    python LMR_surfer.py                        # most recent available date
    python LMR_surfer.py --date 2019-05-15      # specific historical date
    python LMR_surfer.py --date 2011-05-15 --out profile_2011.png
    python LMR_surfer.py --adjust 01145:-8.5    # manually fix a datum offset
    python LMR_surfer.py --list-dates           # show 10 most recent dates
    python LMR_surfer.py --no-dem               # skip slow DEM sampling

Requirements
------------
    pip install rasterio pyproj matplotlib numpy pandas scipy
    (rasterio and pyproj are optional — approximate fallbacks are used)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Import LMR_doctor dataclasses so pickle can deserialise the artifact
# ──────────────────────────────────────────────────────────────────────────────
try:
    from LMR_doctor import StationModelBundle  # noqa: F401  (needed for pickle)
except ImportError:
    pass   # artifact load will fall back to FALLBACK_MILES gracefully

# ──────────────────────────────────────────────────────────────────────────────
# Optional heavy dependencies — graceful degradation
# ──────────────────────────────────────────────────────────────────────────────
try:
    import rasterio  # type: ignore
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from pyproj import Transformer  # type: ignore
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
DEM_PATH      = ROOT / "LMR_DEM.tif"
RIVER_MILES_FILE = ROOT / "river_miles.csv"
OUTPUT_DIR    = ROOT / "models" / "qaqc_outputs"
MODEL_DIR     = ROOT / "models" / "qaqc"
CACHE_DIR     = ROOT / "cache" / "qaqc"
ARTIFACT_PATH = MODEL_DIR / "network_qaqc_artifact.pkl"
CORRECTED_CSV = OUTPUT_DIR / "corrected_stage_network.csv"
RAW_CSV       = OUTPUT_DIR / "raw_stage_network.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Physical / unit constants
# ──────────────────────────────────────────────────────────────────────────────
FT_TO_M  = 0.3048
M_TO_FT  = 1.0 / FT_TO_M
MI_TO_FT = 5280.0          # river miles → feet
FT3_PER_AF = 43_560.0      # cubic feet per acre-foot
FT2_TO_YD2 = 1.0 / 9.0     # square feet → square yards

# DEM coordinate system: NAD83(2011) UTM Zone 15N (EPSG 6344)
DEM_EPSG       = 6344
DEM_PIXEL_M    = 30.0       # metres per pixel (from .tfw)
DEM_UL_EAST    = 586_736.0  # upper-left centre easting (UTM-15N m)
DEM_UL_NORTH   = 3_443_682.0

# DEM cross-section sampling
TRANSECT_HALF_WIDTH_M = 2_500.0   # ±2.5 km either side of centreline
TRANSECT_STEP_M       = 30.0      # matches DEM resolution

# Discharge quantile bin edges (matching LMR_doctor)
DISCHARGE_BIN_LABELS  = ["low", "medium_low", "medium_high", "high"]
DISCHARGE_Q_CUTS      = [0.0, 0.25, 0.50, 0.75, 1.00]

# ──────────────────────────────────────────────────────────────────────────────
# Fallback station river miles (used when the artifact pkl is absent)
# Approximate USACE Lower Mississippi river miles (AHP above Head of Passes)
# ──────────────────────────────────────────────────────────────────────────────
FALLBACK_MILES: Dict[str, float] = {
    "01080": 314.0,   # Red River Landing / Melville LA
    "01120": 302.0,   # Knox Landing LA
    "01145": 261.0,   # St. Francisville / Angola area LA
    "01160": 228.0,   # Baton Rouge gauge vicinity
    "01220": 174.0,   # Baton Rouge (Bayou Manchac reach) LA
    "01240": 158.0,   # Donaldsonville / Darrow LA
    "01280": 127.0,   # Reserve / LaPlace LA
    "01300": 103.0,   # Carrollton / New Orleans LA
    "01390":  63.0,   # Chalmette / Arabi LA
    "01400":  49.0,   # Alliance / Belle Chasse LA
    "01440":  30.0,   # Pointe à la Hache LA
    "01480":  11.0,   # Empire LA
    "01515":   7.0,   # Venice LA
    "01545":   0.0,   # Head of Passes LA
    "01670":   0.0,   # Southwest Pass (below HOP)
}

# USGS discharge stations searched in priority order for the Q proxy
PRIMARY_USGS_Q = ["07374000", "07289000", "07032000"]  # BR, Vicksburg, Memphis

# Main-stem USGS stations with approximate LMR river miles (AHP above Head of
# Passes).  Used to build a spatially-varying discharge profile showing that
# flood waves arrive at different points along the river at different times.
USGS_Q_MAINSTEM: Dict[str, Tuple[str, float]] = {
    "07010000": ("St. Louis MO",   1049.0),
    "07020500": ("Chester IL",       954.0),
    "07022000": ("Thebes IL",        951.0),
    "07032000": ("Memphis TN",       736.0),
    "07289000": ("Vicksburg MS",     437.0),
    "07374000": ("Baton Rouge LA",   228.0),   # primary reference gage
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("LMR_surfer")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_artifact() -> Optional[dict]:
    """Load the LMR_doctor trained artifact (rg_miles, rg_coords, …)."""
    if not ARTIFACT_PATH.exists():
        log.warning("Artifact not found — using fallback station miles")
        return None
    try:
        with ARTIFACT_PATH.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        log.warning("Artifact load failed: %s", exc)
        return None


def load_corrected_stage() -> pd.DataFrame:
    """Return the corrected stage network as a date-indexed DataFrame."""
    if not CORRECTED_CSV.exists():
        raise FileNotFoundError(
            f"Corrected stage file missing: {CORRECTED_CSV}\n"
            "Run `python LMR_doctor.py train` first."
        )
    df = pd.read_csv(CORRECTED_CSV, parse_dates=["date"]).set_index("date")
    df.sort_index(inplace=True)
    return df


def load_raw_stage() -> Optional[pd.DataFrame]:
    """Return the raw (un-corrected) stage network CSV, or None if absent."""
    if not RAW_CSV.exists():
        return None
    try:
        df = pd.read_csv(RAW_CSV, parse_dates=["date"]).set_index("date")
        df.sort_index(inplace=True)
        return df
    except Exception:
        return None


def load_all_discharge_series() -> Dict[str, pd.Series]:
    """
    Load every available USGS discharge series from the cache directory.
    Returns a dict mapping USGS station ID → daily cfs Series.
    Also stores the primary series under key 'primary' for convenience.
    """
    all_q: Dict[str, pd.Series] = {}
    for path in sorted(CACHE_DIR.glob("usgs_q_*_*.csv")):
        # Filename pattern: usgs_q_<id>_<start>_<end>.csv
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        usgs_id = parts[2]          # e.g. "07374000"
        try:
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
            s  = df["value"].dropna()
            if not s.empty:
                all_q[usgs_id] = s
        except Exception as exc:
            log.debug("Could not load %s: %s", path.name, exc)

    # Tag the primary reference series
    for usgs_id in PRIMARY_USGS_Q:
        if usgs_id in all_q:
            all_q["primary"] = all_q[usgs_id]
            log.info("Primary discharge: %s  (%d records)",
                     usgs_id, len(all_q["primary"]))
            break
    else:
        log.warning("No primary USGS discharge found in cache")
        all_q["primary"] = pd.Series(dtype=float)

    log.info("Loaded %d USGS discharge series from cache", len(all_q) - 1)
    return all_q


def load_discharge_series() -> pd.Series:
    """
    Load the best available daily USGS discharge series for the LMR main stem.
    Returns a daily Series in cfs (empty if no cache files found).
    """
    for usgs_id in PRIMARY_USGS_Q:
        for path in sorted(CACHE_DIR.glob(f"usgs_q_{usgs_id}_*.csv")):
            try:
                df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
                s = df["value"].dropna()
                if not s.empty:
                    log.info("Discharge: %s  (%d records)", path.name, len(s))
                    return s
            except Exception as exc:
                log.warning("Could not load %s: %s", path.name, exc)
    log.warning("No USGS discharge found in cache — hysteresis bins unavailable")
    return pd.Series(dtype=float)


def get_station_miles(artifact: Optional[dict]) -> Dict[str, float]:
    if artifact and artifact.get("rg_miles"):
        return dict(artifact["rg_miles"])
    return dict(FALLBACK_MILES)


def get_station_coords(artifact: Optional[dict]) -> Dict[str, Dict[str, float]]:
    if artifact and artifact.get("rg_coords"):
        return dict(artifact["rg_coords"])
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# RIVER CENTRELINE
# ══════════════════════════════════════════════════════════════════════════════

def load_lmr_centreline() -> pd.DataFrame:
    """
    Return the Lower Mississippi River centreline from river_miles.csv.
    Filters RIVER_CODE='MI' and RIVER_NAME containing 'LO' (Lower Mississippi).
    Sorted from upstream (high RM) to downstream (low RM).
    """
    if not RIVER_MILES_FILE.exists():
        raise FileNotFoundError(f"river_miles.csv not found: {RIVER_MILES_FILE}")

    df = pd.read_csv(RIVER_MILES_FILE)
    mask = (
        (df["RIVER_CODE"] == "MI") &
        (df["RIVER_NAME"].str.contains("LO", case=False, na=False))
    )
    lmr = df.loc[mask, ["MILE", "LATITUDE1", "LONGITUDE1"]].copy()
    lmr["MILE"] = pd.to_numeric(lmr["MILE"], errors="coerce")
    lmr = (lmr.dropna(subset=["MILE", "LATITUDE1", "LONGITUDE1"])
               .sort_values("MILE", ascending=False)
               .drop_duplicates(subset="MILE")
               .reset_index(drop=True))
    log.info(
        "LMR centreline: %d river-mile points  RM %.1f → RM %.1f",
        len(lmr), lmr["MILE"].max(), lmr["MILE"].min(),
    )
    return lmr


def compute_river_bearing(centreline: pd.DataFrame, idx: int) -> float:
    """
    Downstream bearing (°CW from north) at centreline point *idx*.
    Uses ±2 neighbouring points for stability.
    """
    n = len(centreline)
    i0, i1 = max(0, idx - 2), min(n - 1, idx + 2)
    lat0, lon0 = centreline["LATITUDE1"].iloc[i0], centreline["LONGITUDE1"].iloc[i0]
    lat1, lon1 = centreline["LATITUDE1"].iloc[i1], centreline["LONGITUDE1"].iloc[i1]
    mean_lat = np.radians((lat0 + lat1) / 2)
    dlat = lat1 - lat0
    dlon = (lon1 - lon0) * np.cos(mean_lat)
    return float(np.degrees(np.arctan2(dlon, dlat)) % 360)


def haversine_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in statute miles."""
    R = 3_958.8
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


# ══════════════════════════════════════════════════════════════════════════════
# DEM PROFILER
# ══════════════════════════════════════════════════════════════════════════════

class DEMProfiler:
    """
    Reads LMR_DEM.tif (NAD83-2011 UTM Zone 15N, metres) and provides
    channel-bed and cross-section queries in feet NAVD88.

    Falls back gracefully if rasterio / pyproj are absent.
    """

    def __init__(self, dem_path: Path = DEM_PATH):
        self.dem_path   = dem_path
        self.available  = False
        self._data: Optional[np.ndarray] = None
        self._transform = None        # rasterio Affine or None
        self._width     = 0
        self._height    = 0
        self._nodata    = -9999.0
        self._to_utm: Optional[object] = None   # pyproj Transformer

        if dem_path.exists() and HAS_RASTERIO:
            self._open()
        elif not dem_path.exists():
            log.warning("DEM not found at %s — channel-bed profile skipped", dem_path)
        else:
            log.warning("rasterio not installed — DEM unavailable.  pip install rasterio")

    def _open(self) -> None:
        try:
            ds = rasterio.open(self.dem_path)
            arr = ds.read(1).astype(np.float64)
            nd  = ds.nodata
            self._nodata = float(nd) if nd is not None else -9999.0
            arr[arr == self._nodata] = np.nan
            arr[arr < -100]          = np.nan   # floor: ~-30 m in ft; deeper = artifact
            arr[arr > 1600]          = np.nan   # ceiling: ~500 m in ft (LMR basin max)
            self._data      = arr
            self._transform = ds.transform
            self._width     = ds.width
            self._height    = ds.height
            self.available  = True
            ds.close()
            log.info(
                "DEM loaded: %d×%d pixels at %.0f m resolution",
                self._width, self._height, DEM_PIXEL_M,
            )
            if HAS_PYPROJ:
                self._to_utm = Transformer.from_crs(
                    "EPSG:4326", f"EPSG:{DEM_EPSG}", always_xy=True
                )
        except Exception as exc:
            log.warning("DEM open failed: %s", exc)

    # ── Coordinate transforms ──────────────────────────────────────────────

    def _latlon_to_utm(self, lat: float, lon: float) -> Tuple[float, float]:
        """WGS84 → UTM Zone 15N easting, northing (metres)."""
        if self._to_utm is not None:
            return self._to_utm.transform(lon, lat)
        # Approximate UTM-15N conversion (central meridian −93°)
        lat_r, lon_r, lon0_r = np.radians(lat), np.radians(lon), np.radians(-93.0)
        a, k0, e2 = 6_378_137.0, 0.9996, 0.006_694_38
        N = a / np.sqrt(1 - e2 * np.sin(lat_r) ** 2)
        T = np.tan(lat_r) ** 2
        C = (e2 / (1 - e2)) * np.cos(lat_r) ** 2
        A = np.cos(lat_r) * (lon_r - lon0_r)
        M = a * (
            (1 - e2 / 4 - 3 * e2 ** 2 / 64) * lat_r
            - (3 * e2 / 8 + 3 * e2 ** 2 / 32) * np.sin(2 * lat_r)
            + (15 * e2 ** 2 / 256) * np.sin(4 * lat_r)
        )
        east  = k0 * N * (A + (1 - T + C) * A ** 3 / 6) + 500_000.0
        north = k0 * (M + N * np.tan(lat_r) * (A ** 2 / 2 + (5 - T + 9 * C) * A ** 4 / 24))
        return float(east), float(north)

    def _utm_to_px(self, east: float, north: float) -> Tuple[int, int]:
        """UTM easting/northing → raster (row, col)."""
        if self._transform is not None:
            t = self._transform
            col = (east  - t.c) / t.a
            row  = (north - t.f) / t.e
        else:
            col = (east  - DEM_UL_EAST)  / DEM_PIXEL_M
            row  = (north - DEM_UL_NORTH) / (-DEM_PIXEL_M)
        return int(round(row)), int(round(col))

    def _px_val(self, row: int, col: int) -> float:
        if self._data is None:
            return np.nan
        if row < 0 or row >= self._height or col < 0 or col >= self._width:
            return np.nan
        v = float(self._data[row, col])
        return np.nan if np.isnan(v) else v

    # ── Public sampling API ────────────────────────────────────────────────

    def sample_thalweg_ft(
        self,
        lat: float,
        lon: float,
        buffer_m: float = 450.0,
    ) -> float:
        """
        Return the 5th-percentile DEM elevation (feet NAVD88) within
        *buffer_m* of the centreline point, approximating the thalweg.
        """
        if not self.available or self._data is None:
            return np.nan
        east, north = self._latlon_to_utm(lat, lon)
        r  = max(1, int(buffer_m / DEM_PIXEL_M))
        vals: List[float] = []
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                rv, cv = self._utm_to_px(east + dc * DEM_PIXEL_M,
                                         north - dr * DEM_PIXEL_M)
                v = self._px_val(rv, cv)
                if not np.isnan(v):
                    vals.append(v)
        return float(np.nanpercentile(vals, 5)) if vals else np.nan   # DEM already in ft NAVD88

    def sample_thalweg_array_ft(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        buffer_m: float = 450.0,
    ) -> np.ndarray:
        """Batch version of sample_thalweg_ft — vectorised over many points."""
        result = np.full(len(lats), np.nan)
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            result[i] = self.sample_thalweg_ft(lat, lon, buffer_m)
        return result

    def transect_wet_area_ft2(
        self,
        lat: float,
        lon: float,
        bearing_deg: float,
        water_surface_ft: float,
        half_width_m: float = TRANSECT_HALF_WIDTH_M,
        step_m: float      = TRANSECT_STEP_M,
    ) -> float:
        """
        Wet cross-sectional area (ft²) from a perpendicular DEM transect.

        Wet area = Σ max(0, water_surface − bed) × cell_width
        for all cells below the water surface.
        """
        if not self.available or self._data is None or np.isnan(water_surface_ft):
            return np.nan
        east0, north0 = self._latlon_to_utm(lat, lon)
        # Perpendicular bearing = downstream bearing + 90°
        perp_rad = np.radians(bearing_deg + 90.0)
        de, dn = np.sin(perp_rad), np.cos(perp_rad)
        offsets = np.arange(-half_width_m, half_width_m + step_m, step_m)
        step_ft = step_m * M_TO_FT
        wet_area = 0.0
        for off in offsets:
            row, col = self._utm_to_px(east0 + de * off, north0 + dn * off)
            bed_ft = self._px_val(row, col)
            if not np.isnan(bed_ft):
                depth_ft = water_surface_ft - bed_ft   # both already in ft NAVD88
                if depth_ft > 0:
                    wet_area += depth_ft * step_ft
        return wet_area


# ══════════════════════════════════════════════════════════════════════════════
# HYDRAULIC PROFILE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class HydraulicProfiler:
    """
    Builds the longitudinal water-surface and channel-bed profile for a
    given date by:
      1. Looking up the corrected stage at each monitored gage.
      2. Linearly interpolating to all centreline river-mile points.
      3. Sampling the DEM thalweg elevation at each centreline point.
    """

    def __init__(
        self,
        corrected_stage: pd.DataFrame,
        station_miles:   Dict[str, float],
        centreline:      pd.DataFrame,
        dem:             DEMProfiler,
    ) -> None:
        self.corrected     = corrected_stage
        self.station_miles = station_miles
        self.centreline    = centreline
        self.dem           = dem
        self._bed_profile_cache: Optional[np.ndarray] = None   # precomputed, reused

        # Build sorted list of (river_mile, station_id) for gages that have data
        stage_cols = set(corrected_stage.columns)
        self._sorted: List[Tuple[float, str]] = sorted(
            [(m, sid) for sid, m in station_miles.items()
             if f"stage_{sid}" in stage_cols],
            key=lambda x: x[0],
            reverse=True,   # upstream first (high RM → low RM)
        )

    # ── Stage at target date ───────────────────────────────────────────────

    def get_stage_at_date(self, target_date: pd.Timestamp) -> Dict[str, float]:
        """
        Return {station_id: stage_ft} for the nearest date to *target_date*.
        Raises ValueError if no date within 7 days is found.
        """
        idx = self.corrected.index
        if target_date in idx:
            row = self.corrected.loc[target_date]
        else:
            nearest = idx[(idx - target_date).abs().argmin()]
            gap = abs((nearest - target_date).days)
            if gap > 7:
                raise ValueError(
                    f"No stage data within 7 days of {target_date.date()}.  "
                    f"Data spans {idx.min().date()} – {idx.max().date()}"
                )
            log.info("Nearest date to %s: %s (%+d days)",
                     target_date.date(), nearest.date(), (nearest - target_date).days)
            row = self.corrected.loc[nearest]

        stages: Dict[str, float] = {}
        for _, sid in self._sorted:
            v = row.get(f"stage_{sid}", np.nan)
            if pd.notna(v):
                stages[sid] = float(v)
        return stages

    # ── Water-surface interpolation ────────────────────────────────────────

    def interpolate_to_miles(
        self,
        stage_at_gages: Dict[str, float],
        target_miles:   np.ndarray,
    ) -> np.ndarray:
        """
        Piecewise-linear interpolation of gage stages to arbitrary river miles.
        Extends as flat (nearest-neighbor) beyond the gage network bounds.
        """
        km = np.array([m for m, sid in self._sorted if sid in stage_at_gages])
        ks = np.array([stage_at_gages[sid] for _, sid in self._sorted
                       if sid in stage_at_gages])
        if len(km) == 0:
            return np.full_like(target_miles, np.nan, dtype=float)
        if len(km) == 1:
            return np.full_like(target_miles, ks[0], dtype=float)
        # np.interp requires increasing x; km is currently high→low
        sort_idx = np.argsort(km)
        km, ks = km[sort_idx], ks[sort_idx]
        return np.interp(target_miles, km, ks)

    # ── Build full profile ─────────────────────────────────────────────────

    def _check_monotonicity(
        self,
        stages: Dict[str, float],
        spike_threshold_ft: float = 5.0,
        envelope_at_q: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Detect positive stage spikes that are hydraulically impossible.

        Rather than enforcing strict downstream monotonicity (which fails on
        reaches like New Orleans where the channel narrows and stage can
        legitimately exceed an upstream wide-floodplain section), this method
        computes the expected stage at each gage by linear interpolation
        between its nearest upstream and downstream neighbours.  A gage is
        flagged only when its stage exceeds that expectation by more than 
        spike_threshold_ft, indicating a sensor or datum error.

        Returns:
            clean_stages  – stages with suspect gages removed (for interpolation)
            flagged       – {sid: suspect_stage} removed (for plotting)
        """
        # upstream-first (descending RM), only gages present in stages
        ordered = [(m, sid) for m, sid in self._sorted if sid in stages]

        if len(ordered) < 2:
            return dict(stages), {}

        flagged: Dict[str, float] = {}
        clean: Dict[str, float]   = dict(stages)

        # Iterative: removing one spike may reveal another, so repeat until stable
        changed = True
        while changed:
            changed = False
            remaining = [(m, sid) for m, sid in ordered if sid in clean]

            for idx, (rm_i, sid) in enumerate(remaining):
                s = clean[sid]

                upstream   = remaining[:idx]   # list of (rm, sid), high-RM first
                downstream = remaining[idx+1:] # list of (rm, sid), low-RM first

                if upstream and downstream:
                    rm_up, s_up = upstream[-1][0],  clean[upstream[-1][1]]
                    rm_dn, s_dn = downstream[0][0], clean[downstream[0][1]]
                    span = rm_up - rm_dn
                    frac = (rm_up - rm_i) / span if span else 0.5
                    expected = s_up + (s_dn - s_up) * frac
                else:
                    # Edge gage — no pair of neighbours to interpolate between;
                    # skip spike detection to avoid false positives.
                    continue

                deviation = s - expected
                if deviation > spike_threshold_ft:
                    log.warning(
                        "SUSPECT stage at %s (RM %.1f): %.1f ft  "
                        "(expected ~%.1f ft, +%.1f ft above profile) "
                        "— excluded from interpolation",
                        sid, rm_i, s, expected, deviation,
                    )
                    flagged[sid] = s
                    del clean[sid]
                    changed = True
                    break   # restart with updated clean set

        # ── P95 envelope exceedance check ─────────────────────────────────────
        # Flag any remaining gage whose stage exceeds the 95th percentile of the
        # historical stage distribution for the current discharge bin.  This is
        # an independent check from the spatial interpolation above and catches
        # implausible gap-fill predictions that happen to lie close to the
        # (potentially corrupted) spatial trendline.
        # A 1 ft buffer is added above P95 to tolerate genuine high-flow events
        # and avoid false-positive flagging of stations barely above P95.
        P95_BUFFER_FT = 1.0
        if envelope_at_q:
            for sid in list(clean.keys()):
                env = envelope_at_q.get(sid)
                if env is None:
                    continue
                s = clean[sid]
                p95 = env.get("p95", float("inf"))
                if s > p95 + P95_BUFFER_FT:
                    rm_i = next((m for m, s2 in self._sorted if s2 == sid), 0.0)
                    log.warning(
                        "P95 EXCEEDANCE at %s (RM %.1f): %.1f ft > historical "
                        "P95=%.1f ft + %.0f ft buffer (Q-bin %.0f–%.0f cfs) "
                        "— excluded from interpolation",
                        sid, rm_i, s, p95, P95_BUFFER_FT,
                        env.get("q_lo", 0), env.get("q_hi", 0),
                    )
                    flagged[sid] = s
                    del clean[sid]

        if flagged:
            log.warning(
                "%d gage(s) flagged as hydraulically inconsistent: %s",
                len(flagged), list(flagged.keys()),
            )
        return clean, flagged

    def build_profile(
        self,
        target_date: pd.Timestamp,
        spike_threshold_ft: float = 5.0,
        manual_adjustments: Optional[Dict[str, float]] = None,
        envelope_at_q: Optional[Dict[str, Dict]] = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with one row per centreline point containing:
          river_mile, lat, lon, bearing_deg,
          bed_elev_ft, water_surface_ft, depth_ft,
          is_gage, gage_id, gage_stage_ft, gage_river_mile
        """
        stages = self.get_stage_at_date(target_date)
        
        # Apply manual datum adjustments before interpolation
        if manual_adjustments:
            for sid, offset in manual_adjustments.items():
                if sid in stages:
                    stages[sid] += offset
                    
        stages, self.flagged_gages = self._check_monotonicity(
            stages, spike_threshold_ft, envelope_at_q=envelope_at_q
        )
        cl     = self.centreline

        rm_vals    = cl["MILE"].values
        lats       = cl["LATITUDE1"].values
        lons       = cl["LONGITUDE1"].values

        # Water surface interpolated to every centreline mile
        ws_profile = self.interpolate_to_miles(stages, rm_vals)

        # DEM thalweg — sampled once for the whole centreline, then smoothed.
        # The bed never changes between dates; cache it for animation performance.
        if self._bed_profile_cache is not None:
            bed_profile = self._bed_profile_cache
        else:
            log.info("Sampling DEM channel-bed elevation at %d centreline points …", len(cl))
            bed_profile = self.dem.sample_thalweg_array_ft(lats, lons, buffer_m=450.0)
            # Smooth the spiky raw thalweg with a rolling median (window = 9 mi)
            # then a 7-point rolling mean to avoid abrupt jumps.
            _bed_s = pd.Series(bed_profile)
            bed_profile = (
                _bed_s.rolling(window=9,  center=True, min_periods=3).median()
                      .rolling(window=7,  center=True, min_periods=3).mean()
                      .values
            )
            self._bed_profile_cache = bed_profile

        # Gage lookup: match each centreline RM to the nearest gage within ±2 mi
        gage_by_mile: Dict[float, Tuple[str, float]] = {
            m: (sid, stages[sid])
            for m, sid in self._sorted
            if sid in stages
        }

        rows: List[dict] = []
        for i in range(len(cl)):
            rm_i   = float(rm_vals[i])
            ws_ft  = float(ws_profile[i])
            bed_ft = float(bed_profile[i])

            bearing = compute_river_bearing(cl, i)

            # Nearest gage within ±2 river miles
            nearest_id, nearest_stage, nearest_gm = "", np.nan, np.nan
            for gm, (gsid, gstg) in gage_by_mile.items():
                if abs(gm - rm_i) < 2.0:
                    nearest_id    = gsid
                    nearest_stage = gstg
                    nearest_gm    = gm
                    break

            rows.append({
                "river_mile":      rm_i,
                "lat":             float(lats[i]),
                "lon":             float(lons[i]),
                "bearing_deg":     bearing,
                "bed_elev_ft":     bed_ft,
                "water_surface_ft": ws_ft,
                "depth_ft":        (ws_ft - bed_ft)
                                   if (np.isfinite(ws_ft) and np.isfinite(bed_ft))
                                   else np.nan,
                "is_gage":         nearest_id != "",
                "gage_id":         nearest_id,
                "gage_stage_ft":   nearest_stage,
                "gage_river_mile": nearest_gm,
            })

        df = pd.DataFrame(rows)
        df.sort_values("river_mile", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# CHANNEL STORAGE CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

class StorageCalculator:
    """
    Computes cumulative in-channel storage (acre-feet) by integrating
    DEM cross-sectional wet areas along the river using the trapezoidal rule.

    dV_i = 0.5 × (A_{i-1} + A_i) × reach_length_ft
    """

    def __init__(self, dem: DEMProfiler) -> None:
        self.dem = dem

    def compute(
        self,
        profile:       pd.DataFrame,
        half_width_m:  float = TRANSECT_HALF_WIDTH_M,
        gage_only:     bool  = False,
    ) -> pd.DataFrame:
        """
        Add ``wet_area_ft2``, ``reach_length_mi``, ``cum_storage_af`` columns
        to a copy of *profile*.

        Parameters
        ----------
        gage_only : If True, only compute wet area at gage stations (faster).
                    Intermediate points will be interpolated.
        """
        n   = len(profile)
        lats = profile["lat"].values
        lons = profile["lon"].values

        # Reach lengths between successive centreline points
        reach_mi = np.zeros(n)
        for i in range(n - 1):
            reach_mi[i] = haversine_mi(lats[i], lons[i], lats[i + 1], lons[i + 1])
        reach_mi[-1] = reach_mi[-2] if n > 1 else 0.0

        # Cross-sectional wet areas
        wet_areas = np.full(n, np.nan)
        compute_mask = (
            profile["is_gage"].values
            if (gage_only and not all(~profile["is_gage"]))
            else np.ones(n, dtype=bool)
        )

        if self.dem.available:
            log.info("Computing DEM cross-section wet areas …")
            for i in range(n):
                if not compute_mask[i]:
                    continue
                ws_ft = profile["water_surface_ft"].iloc[i]
                if not np.isfinite(ws_ft):
                    continue
                bearing = profile["bearing_deg"].iloc[i]
                area = self.dem.transect_wet_area_ft2(
                    lats[i], lons[i], bearing, ws_ft, half_width_m
                )
                wet_areas[i] = area

            # Interpolate areas at non-gage points when gage_only=True
            valid = np.where(np.isfinite(wet_areas))[0]
            if len(valid) > 1:
                all_idx = np.arange(n)
                wet_areas = np.interp(all_idx, valid, wet_areas[valid],
                                      left=wet_areas[valid[0]],
                                      right=wet_areas[valid[-1]])
        else:
            # Fallback: estimate area from typical LMR channel geometry.
            # Channel width varies from ~2 000 ft near the Gulf to ~4 500 ft
            # upstream.  Use water surface elevation relative to sea level as a
            # proxy for depth (higher stage generally = more water in channel).
            ws_vals = profile["water_surface_ft"].values
            rm_vals_s = profile["river_mile"].values
            rm_max_s  = float(np.nanmax(rm_vals_s)) if len(rm_vals_s) else 302.0
            for i in range(n):
                ws_ft = ws_vals[i]
                if not np.isfinite(ws_ft):
                    continue
                rm_i = rm_vals_s[i]
                # Typical LMR channel width increases upstream (wider floodplain)
                width_ft = 2_200 + 2_300 * (rm_i / max(rm_max_s, 1.0))
                # Use half the water-surface elevation as a rough depth proxy
                # (actual thalweg ~20–60 ft below surface depending on RM)
                depth_approx = max(ws_ft * 0.4, 0.0)
                wet_areas[i] = depth_approx * width_ft

        # Cumulative storage by trapezoidal rule
        cum_af = np.zeros(n)
        for i in range(1, n):
            a0 = wet_areas[i - 1] if np.isfinite(wet_areas[i - 1]) else 0.0
            a1 = wet_areas[i]     if np.isfinite(wet_areas[i])     else 0.0
            dv_ft3 = 0.5 * (a0 + a1) * (reach_mi[i - 1] * MI_TO_FT)
            cum_af[i] = cum_af[i - 1] + dv_ft3 / FT3_PER_AF

        out = profile.copy()
        out["wet_area_ft2"]   = wet_areas
        out["reach_length_mi"] = reach_mi
        out["cum_storage_af"] = cum_af
        return out

    def compute_stage_envelope_areas(
        self,
        profile:      pd.DataFrame,
        hysteresis:   "HysteresisAnalyzer",
        q_cfs:        float,
        half_width_m: float = TRANSECT_HALF_WIDTH_M,
    ) -> Dict[str, np.ndarray]:
        """
        Compute cross-sectional wet areas (square yards) at the P05, P50, and
        P95 historical stage levels for each gage station in *profile*, then
        interpolate those control-point values across the full profile.

        Returns a dict with keys ``'p05_yd2'``, ``'p50_yd2'``, ``'p95_yd2'``
        whose values are arrays aligned with ``profile`` (length == len(profile)).
        Returns an empty dict when DEM data are unavailable or fewer than two
        gage stations have historical envelopes.
        """
        n = len(profile)
        rm_all = profile["river_mile"].values

        # --- collect control points (one per gage station) ---
        rm_ctrl:  List[float] = []
        p05_ctrl: List[float] = []
        p50_ctrl: List[float] = []
        p95_ctrl: List[float] = []

        rm_max = float(np.nanmax(rm_all)) if len(rm_all) else 302.0

        seen: set = set()
        for _, row in profile.iterrows():
            if not row["is_gage"]:
                continue
            sid = row["gage_id"]
            if not sid or sid in seen:
                continue
            seen.add(sid)

            env = hysteresis.get_envelope(sid, q_cfs)
            if env is None:
                continue

            lat     = float(row["lat"])
            lon     = float(row["lon"])
            bearing = float(row["bearing_deg"])
            rm_i    = float(row["river_mile"])

            if self.dem.available:
                a05 = self.dem.transect_wet_area_ft2(lat, lon, bearing,
                                                     env["p05"], half_width_m)
                a50 = self.dem.transect_wet_area_ft2(lat, lon, bearing,
                                                     env["p50"], half_width_m)
                a95 = self.dem.transect_wet_area_ft2(lat, lon, bearing,
                                                     env["p95"], half_width_m)
            else:
                # Geometric fallback: same formula used in compute()
                width_ft = 2_200 + 2_300 * (rm_i / max(rm_max, 1.0))
                a05 = max(env["p05"] * 0.4, 0.0) * width_ft
                a50 = max(env["p50"] * 0.4, 0.0) * width_ft
                a95 = max(env["p95"] * 0.4, 0.0) * width_ft

            if np.isfinite(a05) and np.isfinite(a50) and np.isfinite(a95):
                rm_ctrl.append(rm_i)
                p05_ctrl.append(a05 * FT2_TO_YD2)
                p50_ctrl.append(a50 * FT2_TO_YD2)
                p95_ctrl.append(a95 * FT2_TO_YD2)

        if len(rm_ctrl) < 2:
            return {}

        # Sort ascending by river mile (np.interp requires increasing xp)
        order    = np.argsort(rm_ctrl)
        rm_c     = np.array(rm_ctrl)[order]
        p05_c    = np.array(p05_ctrl)[order]
        p50_c    = np.array(p50_ctrl)[order]
        p95_c    = np.array(p95_ctrl)[order]

        def _interp(ctrl: np.ndarray) -> np.ndarray:
            return np.interp(rm_all, rm_c, ctrl,
                             left=ctrl[0], right=ctrl[-1])

        return {
            "p05_yd2": _interp(p05_c),
            "p50_yd2": _interp(p50_c),
            "p95_yd2": _interp(p95_c),
        }


# ══════════════════════════════════════════════════════════════════════════════
# HYSTERESIS / HISTORICAL ENVELOPE
# ══════════════════════════════════════════════════════════════════════════════

class HysteresisAnalyzer:
    """
    Builds per-station historical stage-discharge envelopes.

    For each station and each discharge bin (low / medium_low / medium_high /
    high) the P05–P95 stage range is computed from the full historical record.
    At the same discharge, a rising limb will sit above the historical median
    (more in-channel storage) while a falling limb sits below (draining).

    Stage classification (at current Q):
        "rising"  — stage > P60 for that Q bin
        "falling" — stage < P40 for that Q bin
        "normal"  — between P40 and P60
        "unknown" — discharge data not available
    """

    def __init__(
        self,
        corrected_stage:  pd.DataFrame,
        discharge_series: pd.Series,
        station_miles:    Dict[str, float],
    ) -> None:
        self.stage     = corrected_stage
        self.discharge = discharge_series
        self.s_miles   = station_miles
        self._env: Optional[Dict] = None

    def _build_envelopes(self) -> Dict:
        if self.discharge.empty:
            return {}
        q = self.discharge.reindex(self.stage.index).interpolate(limit=7)
        q_valid = q.dropna()
        if q_valid.empty:
            return {}
        cuts = np.quantile(q_valid.values, DISCHARGE_Q_CUTS)

        envelopes: Dict = {}
        for col in self.stage.columns:
            if not col.startswith("stage_"):
                continue
            sid = col.replace("stage_", "")
            s   = self.stage[col].dropna()
            qa  = q.reindex(s.index)
            combo = pd.DataFrame({"s": s, "q": qa}).dropna()
            if len(combo) < 30:
                continue
            per_bin: Dict[str, Dict] = {}
            for i, label in enumerate(DISCHARGE_BIN_LABELS):
                lo, hi = cuts[i], cuts[i + 1]
                mask = (combo["q"] >= lo) & (combo["q"] <= hi)
                sub  = combo.loc[mask, "s"]
                if len(sub) < 10:
                    continue
                per_bin[label] = {
                    "q_lo":  float(lo),
                    "q_hi":  float(hi),
                    "p05":   float(sub.quantile(0.05)),
                    "p25":   float(sub.quantile(0.25)),
                    "p50":   float(sub.quantile(0.50)),
                    "p75":   float(sub.quantile(0.75)),
                    "p95":   float(sub.quantile(0.95)),
                    "n":     int(mask.sum()),
                }
            if per_bin:
                envelopes[sid] = per_bin
        return envelopes

    @property
    def envelopes(self) -> Dict:
        if self._env is None:
            self._env = self._build_envelopes()
        return self._env

    def current_bin(self, sid: str, q_cfs: float) -> Optional[str]:
        """Return the discharge bin label matching *q_cfs*, or None."""
        env = self.envelopes.get(sid, {})
        for label in DISCHARGE_BIN_LABELS:
            e = env.get(label)
            if e and e["q_lo"] <= q_cfs <= e["q_hi"]:
                return label
        return None

    def get_envelope(self, sid: str, q_cfs: float) -> Optional[Dict]:
        """Return the envelope statistics for the discharge bin at *q_cfs*."""
        label = self.current_bin(sid, q_cfs)
        if label is None:
            return None
        return self.envelopes.get(sid, {}).get(label)

    def classify_limb(self, sid: str, stage_ft: float, q_cfs: float) -> str:
        """Classify stage as 'rising', 'falling', 'normal', or 'unknown'."""
        env = self.get_envelope(sid, q_cfs)
        if env is None:
            return "unknown"
        spread = max(env["p95"] - env["p05"], 0.05)
        rel    = (stage_ft - env["p50"]) / spread
        if rel > 0.20:
            return "rising"
        if rel < -0.20:
            return "falling"
        return "normal"


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_stage_envelope_profile(
    profile:    pd.DataFrame,
    hysteresis: "HysteresisAnalyzer",
    q_cfs:      float,
) -> Dict[str, np.ndarray]:
    """
    Interpolate historical P05 / P50 / P95 stage (ft NAVD88) along the full
    centreline profile at the given discharge, using each gage station's
    hysteresis envelope as a spatial control point.

    Returns a dict with keys ``p05_ft``, ``p50_ft``, ``p95_ft``.
    Empty dict if fewer than 2 control points are available.
    """
    rm_all = profile["river_mile"].values
    rm_ctrl: List[float] = []
    p05_c:   List[float] = []
    p50_c:   List[float] = []
    p95_c:   List[float] = []

    seen: set = set()
    for _, row in profile.iterrows():
        if not row["is_gage"]:
            continue
        sid = row["gage_id"]
        if not sid or sid in seen:
            continue
        seen.add(sid)
        env = hysteresis.get_envelope(sid, q_cfs)
        if env is None:
            continue
        rm_ctrl.append(float(row["gage_river_mile"]))
        p05_c.append(env["p05"])
        p50_c.append(env["p50"])
        p95_c.append(env["p95"])

    if len(rm_ctrl) < 2:
        return {}

    order = np.argsort(rm_ctrl)
    rm_c  = np.array(rm_ctrl)[order]
    p05   = np.array(p05_c)[order]
    p50   = np.array(p50_c)[order]
    p95   = np.array(p95_c)[order]

    return {
        "p05_ft": np.interp(rm_all, rm_c, p05, left=p05[0],  right=p05[-1]),
        "p50_ft": np.interp(rm_all, rm_c, p50, left=p50[0],  right=p50[-1]),
        "p95_ft": np.interp(rm_all, rm_c, p95, left=p95[0],  right=p95[-1]),
    }


def compute_spatial_q_profile(
    profile_rm:   np.ndarray,
    all_discharge: Dict[str, pd.Series],
    target_date:  pd.Timestamp,
) -> Dict:
    """
    Build a spatially-varying discharge profile along the LMR at *target_date*.

    Uses USGS main-stem gauges as spatial control points.  Each gauge reports
    the actual (instantaneous) discharge on target_date.  Because the flood
    wave is unsteady and travels at finite celerity (~3-5 mph), adjacent gauges
    naturally record different Q values at any snapshot in time — the upstream
    gauge may be rising while the downstream gauge has not yet seen the wave.

    The resulting profile illustrates that a single "Reference" discharge
    (e.g. Baton Rouge) is a snapshot at one location; the hydraulic forcing
    varies continuously in space and must be resolved with multiple gauges.

    Returns a dict with:
        ``rm_ctrl``   : sorted river-mile array of control gauges
        ``q_ctrl``    : Q (cfs) at each control gauge on target_date
        ``labels``    : human-readable label per control gauge
        ``q_profile`` : Q linearly interpolated at every profile point
        ``ref_rm``    : river mile of the primary reference gauge (BR)
        ``ref_q``     : Q at the reference gauge
        ``ref_label`` : descriptive label for the reference gauge
    Returns empty dict if no gauge data are available.
    """
    points: List[Tuple[float, float, str]] = []   # (rm, q_cfs, label)
    ref_rm, ref_q, ref_label = 228.0, np.nan, "Baton Rouge (07374000)"

    for usgs_id, (name, rm) in USGS_Q_MAINSTEM.items():
        series = all_discharge.get(usgs_id)
        if series is None or series.empty:
            continue
        try:
            idx   = int(np.abs((series.index - target_date).total_seconds()).argmin())
            q_val = float(series.iloc[idx])
            if np.isfinite(q_val) and q_val > 0:
                points.append((rm, q_val, f"{name}"))
                if usgs_id == "07374000":
                    ref_q = q_val
        except Exception:
            continue

    if not points:
        return {}

    points.sort(key=lambda t: t[0])   # ascending RM
    rm_ctrl = np.array([p[0] for p in points])
    q_ctrl  = np.array([p[1] for p in points])
    labels  = [p[2] for p in points]

    # Interpolate to profile grid; extrapolate flat at both ends
    if len(points) == 1:
        q_profile = np.full_like(profile_rm, q_ctrl[0], dtype=float)
    else:
        q_profile = np.interp(profile_rm, rm_ctrl, q_ctrl,
                              left=q_ctrl[0], right=q_ctrl[-1])

    if np.isnan(ref_q) and len(q_ctrl) > 0:
        closest  = int(np.argmin(np.abs(rm_ctrl - 228.0)))
        ref_q    = q_ctrl[closest]
        ref_rm   = float(rm_ctrl[closest])
        ref_label = labels[closest]

    return {
        "rm_ctrl":   rm_ctrl,
        "q_ctrl":    q_ctrl,
        "labels":    labels,
        "q_profile": q_profile,
        "ref_rm":    ref_rm,
        "ref_q":     ref_q,
        "ref_label": ref_label,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

# Colour palette
C = {
    "bed":         "#5D4037",    # dark brown  — channel bed
    "water_fill":  "#90CAF9",    # sky blue    — water volume
    "water_line":  "#0D47A1",    # deep blue   — water-surface line
    "gage":        "#FF6F00",    # amber       — gage markers
    "storage":     "#2E7D32",    # forest green — storage fill
    "env_fill":    "#FFF3E0",    # very light orange — envelope band fill
    "env_edge":    "#F57C00",    # orange      — envelope band edge
    "env_med":     "#E65100",    # dark orange — median line
    "rising":      "#C62828",    # deep red    — rising limb
    "falling":     "#1565C0",    # deep blue   — falling limb
    "normal":      "#2E7D32",    # green       — near median
    "unknown":     "#757575",    # grey        — no data
}
LIMB_MARKER = {"rising": "^", "falling": "v", "normal": "D", "unknown": "s"}


def _format_label(sid: str, stage_ft: float) -> str:
    return f"{sid}\n{stage_ft:.1f} ft"


def _get_unique_gage_rows(profile: pd.DataFrame) -> List[dict]:
    """Return one representative row per monitored gage, ordered by river mile."""
    seen: set = set()
    rows: List[dict] = []
    for _, r in profile[profile["is_gage"]].iterrows():
        gid = r["gage_id"]
        if gid and gid not in seen:
            seen.add(gid)
            rows.append(r.to_dict())
    # Sort upstream → downstream (descending river mile)
    rows.sort(key=lambda x: x["gage_river_mile"], reverse=True)
    return rows


def plot_hydraulic_profile(
    profile:          pd.DataFrame,
    storage:          pd.DataFrame,
    stages:           Dict[str, float],
    station_miles:    Dict[str, float],
    hysteresis:       "HysteresisAnalyzer",
    q_cfs:            float,
    target_date:      pd.Timestamp,
    flagged_gages:    Optional[Dict[str, float]] = None,
    stage_env:        Optional[Dict[str, np.ndarray]] = None,
    spatial_q:        Optional[Dict] = None,
    fig_out:          Optional[Path] = None,
    show:             bool = True,
    # Pre-built figure + axes for animation (cleared and redrawn each frame)
    fig_axes:         Optional[Tuple] = None,
    elev_ylim:        Optional[Tuple[float, float]] = None,
    q_ylim:           Optional[Tuple[float, float]] = None,
    area_ylim:        Optional[Tuple[float, float]] = None,
    rc_data:          Optional[Dict[str, pd.DataFrame]] = None,
    rc_limits:        Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    rc_stations:      Optional[List[str]] = None,
) -> Optional[object]:
    """
    Render the two-panel hydraulic profile / cross-section storage figure.

    Panel 1 (upper)
    ---------------
    • DEM channel bed (brown fill)
    • Historical P05–P95 stage envelope (interpolated along profile, teal fill)
    • Historical P50 median stage (dashed teal line)
    • Mean bed elevation (grey dashed horizontal line)
    • Interpolated water surface (blue line + water volume fill)
    • Gage markers coloured by hysteresis limb (▲ rising / ▼ falling / ◆ normal)
    • P05–P95 vertical bars at each gage (limb colour)
    • Secondary right y-axis: spatially-varying Q profile from multiple USGS
      gauges — illustrates that the 'Reference' discharge is location-specific

    Panel 2 (lower)
    ---------------
    • Instantaneous cross-sectional wet area (yd²) — line only
    • Historical P05–P95 wet-area envelope (teal fill)
    • Historical P50 median wet area (dashed teal)
    """
    # ── Colours ────────────────────────────────────────────────────────────
    C_BED    = "#5D4037"
    C_WFILL  = "#90CAF9"
    C_WLINE  = "#0D47A1"
    C_RISING = "#C62828"
    C_FALL   = "#1565C0"
    C_NORM   = "#2E7D32"
    C_UNK    = "#757575"
    C_ENV50  = "#00897B"
    C_ENV_F  = "#B2DFDB"    # P05-P95 stage fill
    C_ENV_E  = "#00695C"    # P05-P95 stage edge
    C_INST   = "#0277BD"    # instantaneous wet-area line
    C_ENV50W = "#00897B"    # P50 wet-area line
    C_ENVFW  = "#B2DFDB"    # P05-P95 wet-area fill
    C_BED_M  = "#9E9E9E"    # mean bed elevation line
    C_Q_LINE = "#7B1FA2"    # spatial Q profile colour
    C_Q_REF  = "#E91E63"    # reference Q horizontal line
    LIMB_COLOR  = {"rising": C_RISING, "falling": C_FALL,
                   "normal": C_NORM,   "unknown": C_UNK}
    LIMB_MARKER = {"rising": "^", "falling": "v",
                   "normal": "D", "unknown": "s"}

    # ── Figure / axes ───────────────────────────────────────────────────────
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    if fig_axes is not None:
        if len(fig_axes) == 3:
            fig, ax1, ax2 = fig_axes
            rc_axes = []
        else:
            fig, ax1, ax2, *rc_axes = fig_axes
        ax1.cla(); ax2.cla()
        for ax in rc_axes:
            ax.cla()
        # Remove old twin axes then recreate
        for other in ax1.get_shared_x_axes().get_siblings(ax1):
            if other not in (ax1, ax2, *rc_axes):
                other.cla()
    else:
        if rc_data is not None:
            fig = plt.figure(figsize=(22, 17), dpi=130)
            fig.patch.set_facecolor("#FAFAFA")
            gs  = fig.add_gridspec(3, 5, height_ratios=[3, 2, 1.5],
                                   hspace=0.35, top=0.91, bottom=0.07,
                                   left=0.06, right=0.94)
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
            rc_axes = [fig.add_subplot(gs[2, i]) for i in range(5)]
        else:
            fig = plt.figure(figsize=(22, 13), dpi=130)
            fig.patch.set_facecolor("#FAFAFA")
            gs  = fig.add_gridspec(2, 1, height_ratios=[3, 2],
                                   hspace=0.06, top=0.91, bottom=0.07,
                                   left=0.06, right=0.94)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            rc_axes = []

    ax1r = ax1.twinx()   # secondary right axis for spatial Q
    ax1.set_zorder(ax1r.get_zorder() + 1)
    ax1.patch.set_visible(False)

    rm  = profile["river_mile"].values
    bed = profile["bed_elev_ft"].values
    ws  = profile["water_surface_ft"].values

    wa_ft2 = (storage["wet_area_ft2"].values
              if "wet_area_ft2" in storage.columns
              else np.full(len(rm), np.nan))
    wa_yd2 = wa_ft2 * FT2_TO_YD2

    has_env_w = all(c in storage.columns
                    for c in ("p05_yd2", "p50_yd2", "p95_yd2"))
    if has_env_w:
        p05_yd2 = storage["p05_yd2"].values
        p50_yd2 = storage["p50_yd2"].values
        p95_yd2 = storage["p95_yd2"].values

    bed_ok = np.isfinite(bed)
    ws_ok  = np.isfinite(ws)
    wa_ok  = np.isfinite(wa_yd2)

    gage_rows = _get_unique_gage_rows(profile)

    # x-axis limits
    if gage_rows:
        x_up = max(float(gr["gage_river_mile"]) for gr in gage_rows) + 14
        x_dn = min(float(gr["gage_river_mile"]) for gr in gage_rows) - 8
    else:
        x_up, x_dn = float(np.nanmax(rm)) + 6, float(np.nanmin(rm)) - 6
    in_view  = (rm >= x_dn) & (rm <= x_up)

    # ═══════════════════════════════════════════════════════════════════
    # PANEL 1 — Elevation profile + stage envelopes + spatial Q
    # ═══════════════════════════════════════════════════════════════════
    ax1.set_facecolor("#EAF2FB")

    # --- Historical P05–P95 stage envelope (continuous fill along profile) ---
    if stage_env:
        p05_s = stage_env.get("p05_ft")
        p50_s = stage_env.get("p50_ft")
        p95_s = stage_env.get("p95_ft")
        env_ok = np.isfinite(p05_s) & np.isfinite(p95_s) if p05_s is not None else np.zeros(len(rm), bool)
        if env_ok.any():
            ax1.fill_between(rm, p05_s, p95_s, where=env_ok,
                             color=C_ENV_F, alpha=0.55, zorder=2,
                             label="Hist. P05–P95 stage (current Q bin)")
            ax1.plot(rm, np.where(np.isfinite(p50_s), p50_s, np.nan),
                     color=C_ENV50, lw=1.5, ls="--", alpha=0.85, zorder=3,
                     label="Hist. P50 (median) stage")

    # --- Channel bed fill ---
    if bed_ok.any():
        y_floor = np.nanmin(bed[bed_ok]) - 15
        ax1.fill_between(rm, y_floor, bed, where=bed_ok,
                         color=C_BED, alpha=0.90, zorder=4,
                         label="Channel bed (DEM thalweg)")
        ax1.plot(rm, bed, color=C_BED, lw=0.9, zorder=5)

    # --- Mean bed elevation (horizontal reference line) ---
    bed_in_view = bed[in_view & bed_ok]
    if bed_in_view.size > 0:
        mean_bed = float(np.nanmean(bed_in_view))
        ax1.axhline(mean_bed, color=C_BED_M, lw=1.2, ls=(0, (5, 3)),
                    alpha=0.80, zorder=6,
                    label=f"Mean thalweg elev. ({mean_bed:.1f} ft NAVD88)")

    # --- Water volume fill + water surface line ---
    if ws_ok.any() and bed_ok.any():
        wt = np.where(ws_ok & bed_ok, ws,  np.nan)
        wb = np.where(ws_ok & bed_ok, bed, np.nan)
        ax1.fill_between(rm, wb, wt,
                         where=np.isfinite(wt) & np.isfinite(wb),
                         color=C_WFILL, alpha=0.60, zorder=7,
                         label="In-channel water volume")
    if ws_ok.any():
        ax1.plot(rm, ws, color=C_WLINE, lw=2.0, zorder=8,
                 label="Water surface (interpolated)")

    # --- Gage markers: P05–P95 bars + limb-coloured marker + annotation ---
    for i, gr in enumerate(gage_rows):
        sid   = gr["gage_id"]
        gm    = float(gr["gage_river_mile"])
        gs_ft = float(gr["gage_stage_ft"])
        env   = hysteresis.get_envelope(sid, q_cfs)
        limb  = hysteresis.classify_limb(sid, gs_ft, q_cfs)
        lc    = LIMB_COLOR.get(limb, C_UNK)
        lmk   = LIMB_MARKER.get(limb, "s")

        for ax_g in (ax1, ax2):
            ax_g.axvline(gm, color="#9E9E9E", lw=0.5, ls=":", alpha=0.35, zorder=0)

        if env:
            ax1.vlines(gm, env["p05"], env["p95"],
                       colors=lc, lw=4.0, alpha=0.28, zorder=9)
            ax1.vlines(gm, env["p25"], env["p75"],
                       colors=lc, lw=7.5, alpha=0.42, zorder=10)
            ax1.hlines(env["p50"], gm - 1.4, gm + 1.4,
                       colors=lc, lw=2.2, alpha=0.92, zorder=11)

        ax1.scatter(gm, gs_ft, s=110, color=lc, marker=lmk,
                    zorder=12, edgecolors="white", linewidths=0.8)

        if env:
            delta    = gs_ft - env["p50"]
            ann_text = f"{sid}\n{gs_ft:.1f} ft\n\u0394mdn: {delta:+.1f} ft"
        else:
            ann_text = f"{sid}\n{gs_ft:.1f} ft"

        ax1.annotate(
            ann_text,
            xy=(gm, gs_ft),
            xytext=(0, 18 + (i % 4) * 14),
            textcoords="offset points",
            fontsize=7.5, ha="center", va="bottom", color=lc,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=lc, alpha=0.90, lw=0.7),
            arrowprops=dict(arrowstyle="-", color=lc, lw=0.5),
            zorder=13,
        )

    # --- Suspect gages ---
    if flagged_gages:
        for sid, s_ft in flagged_gages.items():
            rm_f = station_miles.get(sid)
            if rm_f is None:
                continue
            ax1.scatter(rm_f, s_ft, s=140, color="crimson", marker="X",
                        zorder=14, edgecolors="white", linewidths=0.8)
            ax1.annotate(
                f"{sid}\n{s_ft:.1f} ft\n[SUSPECT]",
                xy=(rm_f, s_ft), xytext=(0, -38),
                textcoords="offset points",
                fontsize=7.0, ha="center", va="top", color="crimson",
                bbox=dict(boxstyle="round,pad=0.25", fc="#FFF0F0",
                          ec="crimson", alpha=0.92, lw=0.9),
                arrowprops=dict(arrowstyle="-", color="crimson", lw=0.7),
                zorder=15,
            )

    # --- Secondary axis: spatially-varying Q profile ---
    ax1r.set_ylabel("Discharge  (cfs)", fontsize=9, color=C_Q_LINE, labelpad=6)
    ax1r.tick_params(axis="y", labelcolor=C_Q_LINE, labelsize=8)
    ax1r.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e3:.0f}k"))
    ax1r.spines["right"].set_color(C_Q_LINE)

    sq_drawn = False
    if spatial_q and "q_profile" in spatial_q:
        sq_rm  = profile["river_mile"].values
        sq_q   = spatial_q["q_profile"]
        sq_ok  = np.isfinite(sq_q) & (sq_rm >= x_dn) & (sq_rm <= x_up)

        if sq_ok.any():
            ax1r.plot(sq_rm[sq_ok], sq_q[sq_ok],
                      color=C_Q_LINE, lw=1.6, ls="-.", alpha=0.70,
                      zorder=1, label="Spatial Q (multi-gauge)")
            sq_drawn = True

        # Control-point dots
        for rm_i, q_i, lbl in zip(spatial_q["rm_ctrl"],
                                   spatial_q["q_ctrl"],
                                   spatial_q["labels"]):
            if x_dn <= rm_i <= x_up:
                ax1r.scatter(rm_i, q_i, s=55, color=C_Q_LINE,
                             zorder=3, edgecolors="white", linewidths=0.7)
                ax1r.annotate(
                    f"{lbl}\n{q_i/1e3:.0f}k cfs",
                    xy=(rm_i, q_i),
                    xytext=(0, -28),
                    textcoords="offset points",
                    fontsize=6.5, ha="center", va="top", color=C_Q_LINE,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=C_Q_LINE, alpha=0.85, lw=0.6),
                    zorder=4,
                )

        # Reference Q horizontal dashed line
        ref_q   = spatial_q.get("ref_q",   q_cfs)
        ref_lbl = spatial_q.get("ref_label", "Reference")
        ref_rm  = spatial_q.get("ref_rm",   228.0)
        if np.isfinite(ref_q):
            ax1r.axhline(ref_q, color=C_Q_REF, lw=1.4, ls="--",
                         alpha=0.80, zorder=2,
                         label=f"Reference Q — {ref_lbl} RM {ref_rm:.0f}")
            ax1r.annotate(
                f"Ref. Q\n{ref_q/1e3:.0f}k cfs\n(RM {ref_rm:.0f})",
                xy=(x_up - (x_up - x_dn) * 0.05, ref_q),
                fontsize=6.5, ha="right", va="center", color=C_Q_REF,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=C_Q_REF, alpha=0.85, lw=0.6),
            )

    # Q-axis limits (leave headroom so it doesn't crowd the bed/WS)
    q_vals_vis: list = list(spatial_q["q_ctrl"]) if spatial_q and "q_ctrl" in spatial_q else []
    if q_cfs > 0:
        q_vals_vis.append(q_cfs)
    if q_vals_vis:
        # Default behaviour: dynamic headroom. If animation supplied a fixed
        # q_ylim, use it instead (locks y-axis across frames).
        if q_ylim is not None:
            ax1r.set_ylim(q_ylim[0], q_ylim[1])
        else:
            q_max = max(q_vals_vis) * 1.30
            q_min = max(0.0, min(q_vals_vis) * 0.70)
            ax1r.set_ylim(q_min, q_max)

    # ax1 y-limits
    bed_view = bed[in_view & bed_ok]
    ws_view  = ws [in_view & ws_ok]
    y_lo = (float(np.nanmin(bed_view)) - 10) if bed_view.size > 0 else -15
    y_hi = (float(np.nanmax(ws_view))  + 14) if ws_view.size  > 0 else 70
    if stage_env and stage_env.get("p95_ft") is not None:
        p95v = stage_env["p95_ft"][in_view]
        if np.isfinite(p95v).any():
            y_hi = max(y_hi, float(np.nanmax(p95v)) + 4)
    for gr in gage_rows:
        env = hysteresis.get_envelope(gr["gage_id"], q_cfs)
        if env:
            y_hi = max(y_hi, env["p95"] + 4)

    ax1.set_xlim(x_up, x_dn)
    # If caller provided a locked elevation y-range (animation), use it.
    if elev_ylim is not None:
        ax1.set_ylim(elev_ylim[0], elev_ylim[1])
    else:
        ax1.set_ylim(y_lo, y_hi)
    ax1r.set_xlim(x_up, x_dn)
    ax1.set_ylabel("Elevation (ft NAVD88)", fontsize=10, labelpad=6)
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.grid(True, ls="--", lw=0.4, alpha=0.55, color="grey")
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    p1_handles = [
        mpatches.Patch(color=C_ENV_F, alpha=0.60, ec=C_ENV_E,
                       label="Hist. P05–P95 stage (current Q bin)"),
        Line2D([0], [0], color=C_ENV50,  lw=1.5, ls="--",
               label="Hist. P50 (median) stage"),
        mpatches.Patch(color=C_BED, label="Channel bed (DEM thalweg)"),
        Line2D([0], [0], color=C_BED_M,  lw=1.2, ls=(0, (5, 3)),
               label=f"Mean thalweg elev. ({mean_bed:.1f} ft)" if bed_in_view.size else "Mean thalweg"),
        mpatches.Patch(color=C_WFILL, alpha=0.65, label="In-channel water"),
        Line2D([0], [0], color=C_WLINE,  lw=2, label="Water surface"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_RISING, markersize=9,
               label="Rising limb (> P60)"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=C_FALL,   markersize=9,
               label="Falling limb (< P40)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_NORM,   markersize=9,
               label="Near historical median"),
    ]
    p1_handles = []
    if stage_env and env_ok.any():
        p1_handles.append(mpatches.Patch(color=C_ENV_F, alpha=0.60, ec=C_ENV_E, label="Hist. P05–P95 stage (current Q bin)"))
        p1_handles.append(Line2D([0], [0], color=C_ENV50, lw=1.5, ls="--", label="Hist. P50 (median) stage"))
    if bed_ok.any():
        p1_handles.append(mpatches.Patch(color=C_BED, label="Channel bed (DEM thalweg)"))
    if bed_in_view.size > 0:
        p1_handles.append(Line2D([0], [0], color=C_BED_M, lw=1.2, ls=(0, (5, 3)), label=f"Mean thalweg elev. ({mean_bed:.1f} ft)"))
    if ws_ok.any() and bed_ok.any():
        p1_handles.append(mpatches.Patch(color=C_WFILL, alpha=0.65, label="In-channel water"))
    if ws_ok.any():
        p1_handles.append(Line2D([0], [0], color=C_WLINE, lw=2, label="Water surface"))

    present_limbs = {hysteresis.classify_limb(gr["gage_id"], float(gr["gage_stage_ft"]), q_cfs) for gr in gage_rows}
    if "rising" in present_limbs:
        p1_handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor=C_RISING, markersize=9, label="Rising limb (> P60)"))
    if "falling" in present_limbs:
        p1_handles.append(Line2D([0], [0], marker="v", color="w", markerfacecolor=C_FALL, markersize=9, label="Falling limb (< P40)"))
    if "normal" in present_limbs:
        p1_handles.append(Line2D([0], [0], marker="D", color="w", markerfacecolor=C_NORM, markersize=9, label="Near historical median"))

    if sq_drawn:
        p1_handles.append(Line2D([0], [0], color=C_Q_LINE, lw=1.6, ls="-.", label="Spatial Q (multi-gauge, right axis)"))
        p1_handles.append(Line2D([0], [0], color=C_Q_REF, lw=1.4, ls="--", label="Reference Q (right axis)"))

    if flagged_gages:
        p1_handles.append(Line2D([0], [0], marker="X", color="w", markerfacecolor="crimson", markersize=9, label="Suspect reading (excluded)"))

    ax1.legend(handles=p1_handles, loc="upper right",
               fontsize=7.5, framealpha=0.92, edgecolor="#BDBDBD", ncol=2)

    # ═══════════════════════════════════════════════════════════════════
    # PANEL 2 — Cross-sectional wet area (yd²)
    # ═══════════════════════════════════════════════════════════════════
    ax2.set_facecolor("#EBF5FB")

    if wa_ok.any():
        if has_env_w:
            env_ok = np.isfinite(p05_yd2) & np.isfinite(p95_yd2)
            if env_ok.any():
                ax2.fill_between(rm, p05_yd2, p95_yd2, where=env_ok,
                                 color=C_ENVFW, alpha=0.75, zorder=2,
                                 label="Hist. P05–P95 area (current Q bin)")
            ax2.plot(rm, np.where(np.isfinite(p50_yd2), p50_yd2, np.nan),
                     color=C_ENV50W, lw=1.8, ls="--", alpha=0.90, zorder=3,
                     label="Hist. P50 (median) area")

        # Instantaneous wet area — line only (no fill)
        ax2.plot(rm, np.where(wa_ok, wa_yd2, np.nan),
                 color=C_INST, lw=2.2, zorder=4,
                 label="Instantaneous wet area")

        # Gage location ticks on the instantaneous line
        rm_s = storage["river_mile"].values
        for gr in gage_rows:
            idx = int(np.argmin(np.abs(rm_s - float(gr["gage_river_mile"]))))
            if np.isfinite(wa_yd2[idx]):
                ax2.scatter(gr["gage_river_mile"], wa_yd2[idx], s=60,
                            color=C_INST, zorder=6, edgecolors="white", linewidths=0.7)

        all_ya = list(wa_yd2[wa_ok])
        if has_env_w:
            all_ya += [float(v) for v in p95_yd2 if np.isfinite(v)]
        y2_max = max(all_ya) * 1.20 if all_ya else 1.0
        # Allow locked area limits for animation; otherwise use dynamic upper bound
        if area_ylim is not None:
            ax2.set_ylim(area_ylim[0], area_ylim[1])
        else:
            ax2.set_ylim(0, y2_max)
        ax2.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    else:
        ax2.text(0.5, 0.5,
                 "Cross-section data unavailable\n(re-run without --no-dem)",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=11, color="#757575",
                 bbox=dict(boxstyle="round", fc="white", ec="#BDBDBD", alpha=0.8))

    if q_cfs > 0:
        ax2.text(0.01, 0.97,
                 f"Q \u2248 {q_cfs:,.0f} cfs  (Reference — Baton Rouge RM 228)",
                 transform=ax2.transAxes, fontsize=8.5, va="top", color="#37474F",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec="#90A4AE", alpha=0.85))

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel("River Mile   (upstream \u2190 \u2014 \u2192 downstream)",
                   fontsize=10, labelpad=6)
    ax2.set_ylabel("Cross-Sectional Wet Area  (yd\u00b2)", fontsize=10, labelpad=6)
    ax2.grid(True, ls="--", lw=0.4, alpha=0.45, color="grey")
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    p2_handles = [
        Line2D([0], [0], color=C_INST, lw=2.2, label="Instantaneous wet area"),
    ]
    if has_env_w:
        p2_handles += [
            mpatches.Patch(color=C_ENVFW, alpha=0.75,
                           label="Hist. P05–P95 area (current Q bin)"),
            Line2D([0], [0], color=C_ENV50W, lw=1.8, ls="--",
                   label="Hist. P50 (median) area"),
        ]
    ax2.legend(handles=p2_handles, loc="upper right",
               fontsize=8, framealpha=0.92, edgecolor="#BDBDBD")

    # ═══════════════════════════════════════════════════════════════════
    # PANEL 3 — Rating Curves
    # ═══════════════════════════════════════════════════════════════════
    if rc_axes and rc_stations and rc_data is not None:
        for i, sid in enumerate(rc_stations):
            ax_rc = rc_axes[i]
            df_rc = rc_data.get(sid)
            if df_rc is not None and not df_rc.empty:
                ax_rc.plot(df_rc['q'], df_rc['stage'], color='red', alpha=0.4, lw=1.5, zorder=1)
                ax_rc.scatter(df_rc['q'], df_rc['stage'], color='red', s=20, zorder=2)
                ax_rc.scatter(df_rc['q'].iloc[-1], df_rc['stage'].iloc[-1], color='gold', s=60, edgecolors='black', zorder=3)
            
            ax_rc.set_title(f"Station {sid} Rating Curve", fontsize=10)
            ax_rc.set_xlabel("Discharge (cfs)", fontsize=9)
            if i == 0:
                ax_rc.set_ylabel("Stage (ft NAVD88)", fontsize=9)
            
            ax_rc.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
            ax_rc.grid(True, ls="--", alpha=0.4)
            
            if rc_limits and sid in rc_limits:
                q_min, q_max, s_min, s_max = rc_limits[sid]
                ax_rc.set_xlim(q_min, q_max)
                ax_rc.set_ylim(s_min, s_max)

    # ═══════════════════════════════════════════════════════════════════
    # Title + x-axis ticks
    # ═══════════════════════════════════════════════════════════════════
    title_date = (
        target_date.strftime("%B %-d, %Y")
        if sys.platform != "win32"
        else target_date.strftime("%B %d, %Y").replace(" 0", " ")
    )
    q_str = f"Q \u2248 {q_cfs:,.0f} cfs (ref.)" if q_cfs > 0 else "Q unavailable"
    fig.suptitle(
        "Lower Mississippi River  \u2014  Hydraulic Profile & Channel Storage Dynamics\n"
        f"{title_date}    \u00b7    {q_str}    \u00b7    Red River Landing \u2192 Gulf of Mexico",
        fontsize=12.5, fontweight="bold", y=0.975,
    )
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if fig_out:
        fig_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_out, dpi=150, bbox_inches="tight")
        log.info("Figure saved \u2192 %s", fig_out)
    if show:
        plt.show()
    if fig_axes is None:
        plt.close(fig)
    return fig




# ══════════════════════════════════════════════════════════════════════════════
# FRAME BUILDER  (shared by single-date and animation paths)
# ══════════════════════════════════════════════════════════════════════════════

def build_one_frame(
    target_date:         pd.Timestamp,
    corrected:           pd.DataFrame,
    all_discharge:       Dict[str, pd.Series],
    sta_miles:           Dict[str, float],
    centreline:          pd.DataFrame,
    dem:                 "DEMProfiler",
    hysteresis:          "HysteresisAnalyzer",
    storage_calc:        "StorageCalculator",
    profiler:            "HydraulicProfiler",
    spike_thresh:        float = 5.0,
    transect_km:         float = TRANSECT_HALF_WIDTH_M / 1_000,
    manual_adjustments:  Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float],
           float, Dict[str, float], Dict, Dict]:
    """
    Compute all data needed to render one plot frame.

    Returns (profile_df, storage_df, stages, q_cfs, flagged_gages,
             spatial_q_dict, stage_env_dict).
    The profiler is reused across frames so its bed-profile cache is retained.
    """
    discharge = all_discharge.get("primary", pd.Series(dtype=float))

    # Reference discharge at target_date
    q_cfs = 0.0
    if not discharge.empty:
        dt_idx = int(np.abs((discharge.index - target_date).total_seconds()).argmin())
        q_cfs  = float(discharge.iloc[dt_idx])

    # Per-station envelope at current Q (for spike detection and stage profile)
    envelope_at_q: Dict[str, Dict] = {
        sid: env
        for sid in sta_miles
        for env in [hysteresis.get_envelope(sid, q_cfs)]
        if env is not None
    }

    # Hydraulic profile (bed cache reused from profiler)
    profile_df = profiler.build_profile(
        target_date,
        spike_threshold_ft=spike_thresh,
        envelope_at_q=envelope_at_q,
        manual_adjustments=manual_adjustments,
    )
    stages  = profiler.get_stage_at_date(target_date)
    flagged = getattr(profiler, "flagged_gages", {})

    # In-channel storage and historical envelope areas
    half_width_m = transect_km * 1_000
    storage_df   = storage_calc.compute(profile_df, half_width_m=half_width_m,
                                        gage_only=True)
    env_areas = storage_calc.compute_stage_envelope_areas(
        storage_df, hysteresis, q_cfs, half_width_m=half_width_m
    )
    for col, arr in env_areas.items():
        storage_df[col] = arr

    # Spatial Q profile across main-stem gauges
    spatial_q   = compute_spatial_q_profile(
        profile_df["river_mile"].values, all_discharge, target_date
    )
    # Historical stage envelope along the full profile
    stage_env   = compute_stage_envelope_profile(profile_df, hysteresis, q_cfs)

    return profile_df, storage_df, stages, q_cfs, flagged, spatial_q, stage_env


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATION
# ══════════════════════════════════════════════════════════════════════════════

def animate_profile(
    dates:          List[pd.Timestamp],
    corrected:      pd.DataFrame,
    all_discharge:  Dict[str, pd.Series],
    sta_miles:      Dict[str, float],
    centreline:     pd.DataFrame,
    dem:            "DEMProfiler",
    hysteresis:     "HysteresisAnalyzer",
    spike_thresh:   float,
    transect_km:    float,
    manual_adjustments: Optional[Dict[str, float]],
    anim_out:       Path,
    fps:            int = 5,
) -> None:
    """
    Render a time-series animation of the hydraulic profile figure.

    All heavy static data (DEM bed profile, hysteresis envelopes) are
    precomputed on the first frame and reused across subsequent frames via
    the HydraulicProfiler bed-profile cache and the HysteresisAnalyzer
    envelope cache.

    Output format is inferred from *anim_out* extension:
        .gif  → Pillow writer  (no external tools required)
        .mp4  → FFmpeg writer  (requires ffmpeg on PATH)
    """
    log.info("Animation: %d frames  (%s → %s)  fps=%d  out=%s",
             len(dates), dates[0].date(), dates[-1].date(), fps, anim_out)

    storage_calc = StorageCalculator(dem)
    # One shared profiler — its bed cache is populated on the first frame
    profiler = HydraulicProfiler(corrected, sta_miles, centreline, dem)

    rc_stations = ["01080", "01160", "01300", "01400", "01670"]
    rc_limits = {}
    primary_q = all_discharge.get("primary")
    if primary_q is not None:
        mask_all = (corrected.index >= dates[0]) & (corrected.index <= dates[-1])
        df_all = corrected.loc[mask_all]
        q_all = primary_q.reindex(df_all.index).interpolate(limit=7)
        for sid in rc_stations:
            col = f"stage_{sid}"
            if col in df_all.columns:
                df_rc = pd.DataFrame({'stage': df_all[col], 'q': q_all}).dropna()
                if not df_rc.empty:
                    q_min, q_max = df_rc['q'].min(), df_rc['q'].max()
                    s_min, s_max = df_rc['stage'].min(), df_rc['stage'].max()
                    q_buf = (q_max - q_min) * 0.1 if q_max > q_min else q_max * 0.1
                    s_buf = (s_max - s_min) * 0.1 if s_max > s_min else s_max * 0.1
                    if q_buf == 0: q_buf = 1000
                    if s_buf == 0: s_buf = 5
                    rc_limits[sid] = (q_min - q_buf, q_max + q_buf, s_min - s_buf, s_max + s_buf)

    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(22, 17), dpi=110)
    fig.patch.set_facecolor("#FAFAFA")
    gs  = fig.add_gridspec(3, 5, height_ratios=[3, 2, 1.5],
                           hspace=0.35, top=0.91, bottom=0.07,
                           left=0.06, right=0.94)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    rc_axes = [fig.add_subplot(gs[2, i]) for i in range(5)]

    frame_data: List = []    # pre-built data for all frames

    log.info("Pre-computing frame data …")
    for i, dt in enumerate(dates):
        if i % 10 == 0:
            log.info("  Frame %d / %d  (%s)", i + 1, len(dates), dt.date())
        try:
            fd = build_one_frame(
                target_date=dt,
                corrected=corrected,
                all_discharge=all_discharge,
                sta_miles=sta_miles,
                centreline=centreline,
                dem=dem,
                hysteresis=hysteresis,
                storage_calc=storage_calc,
                profiler=profiler,
                spike_thresh=spike_thresh,
                transect_km=transect_km,
                manual_adjustments=manual_adjustments,
            )
            frame_data.append((dt, fd))
        except Exception as exc:
            log.warning("Frame %s failed: %s — skipped", dt.date(), exc)

    if not frame_data:
        log.error("No frames could be computed — animation aborted")
        return

    # Compute global y-limits across all frames so axes can be locked for
    # the entire animation (prevents autoscaling flicker).
    elev_min = float("inf")
    elev_max = float("-inf")
    q_vals_all: List[float] = []
    area_vals_all: List[float] = []
    for _dt, fd in frame_data:
        profile_df, storage_df, stages, q_cfs, flagged, spatial_q, stage_env = fd
        bed_a = profile_df.get("bed_elev_ft")
        ws_a  = profile_df.get("water_surface_ft")
        if bed_a is not None and np.isfinite(bed_a.values).any():
            elev_min = min(elev_min, float(np.nanmin(bed_a.values)))
        if ws_a is not None and np.isfinite(ws_a.values).any():
            elev_max = max(elev_max, float(np.nanmax(ws_a.values)))
        if stage_env and stage_env.get("p95_ft") is not None:
            p95 = np.asarray(stage_env.get("p95_ft"))
            if np.isfinite(p95).any():
                elev_max = max(elev_max, float(np.nanmax(p95)))

        # Q values (spatial profile + control points + reference)
        if spatial_q and isinstance(spatial_q, dict):
            try:
                qp = np.asarray(spatial_q.get("q_profile", []), dtype=float)
                q_vals_all += [float(x) for x in qp[np.isfinite(qp)]]
            except Exception:
                pass
            try:
                qc = spatial_q.get("q_ctrl", [])
                q_vals_all += [float(x) for x in qc if np.isfinite(x)]
            except Exception:
                pass
        # include reference q for this frame
        if np.isfinite(q_cfs):
            q_vals_all.append(float(q_cfs))

        # Area values (instantaneous + envelope)
        if storage_df is not None:
            if "wet_area_ft2" in storage_df.columns:
                wa = np.asarray(storage_df["wet_area_ft2"].values) * FT2_TO_YD2
                area_vals_all += [float(x) for x in wa[np.isfinite(wa)]]
            for col in ("p05_yd2", "p50_yd2", "p95_yd2"):
                if col in storage_df.columns:
                    arr = np.asarray(storage_df[col].values)
                    area_vals_all += [float(x) for x in arr[np.isfinite(arr)]]

    # Finalize limits (use None when no data)
    elev_ylim: Optional[Tuple[float, float]] = None
    q_ylim: Optional[Tuple[float, float]] = None
    area_ylim: Optional[Tuple[float, float]] = None
    if np.isfinite(elev_min) and np.isfinite(elev_max) and elev_min < elev_max:
        elev_ylim = (float(elev_min) - 10.0, float(elev_max) + 14.0)
    if q_vals_all:
        q_min = float(np.nanmin(q_vals_all))
        q_max = float(np.nanmax(q_vals_all))
        if q_min == q_max:
            # avoid zero-range
            q_min = 0.0 if q_min <= 0.0 else q_min * 0.995
            q_max = q_max * 1.005 if q_max != 0.0 else 1.0
        q_ylim = (q_min, q_max)
    if area_vals_all:
        a_min = float(np.nanmin(area_vals_all))
        a_max = float(np.nanmax(area_vals_all))
        if a_min == a_max:
            a_min = 0.0
            a_max = a_max if a_max > 0 else 1.0
        area_ylim = (a_min, a_max)

    def _update(frame_idx: int) -> list:
        dt, fd = frame_data[frame_idx]
        profile_df, storage_df, stages, q_cfs, flagged, spatial_q, stage_env = fd
        
        rc_data = {}
        if primary_q is not None:
            mask_frame = (corrected.index >= dates[0]) & (corrected.index <= dt)
            df_frame = corrected.loc[mask_frame]
            q_frame = primary_q.reindex(df_frame.index).interpolate(limit=7)
            for sid in rc_stations:
                col = f"stage_{sid}"
                if col in df_frame.columns:
                    rc_data[sid] = pd.DataFrame({'stage': df_frame[col], 'q': q_frame}).dropna()

        # Clear old twin axis before plot_hydraulic_profile recreates it
        for _ax in fig.get_axes():
            if _ax not in (ax1, ax2, *rc_axes):
                _ax.remove()
        plot_hydraulic_profile(
            profile       = profile_df,
            storage       = storage_df,
            stages        = stages,
            station_miles = sta_miles,
            hysteresis    = hysteresis,
            q_cfs         = q_cfs,
            target_date   = dt,
            flagged_gages = flagged,
            stage_env     = stage_env,
            spatial_q     = spatial_q,
            fig_out       = None,
            show          = False,
            fig_axes      = (fig, ax1, ax2, *rc_axes),
            elev_ylim     = elev_ylim,
            q_ylim        = q_ylim,
            area_ylim     = area_ylim,
            rc_data       = rc_data,
            rc_limits     = rc_limits,
            rc_stations   = rc_stations,
        )
        return []

    anim = manimation.FuncAnimation(
        fig, _update,
        frames=len(frame_data),
        interval=int(1000 / fps),
        blit=False,
        repeat=False,
    )

    suffix = anim_out.suffix.lower()
    anim_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if suffix == ".gif":
            log.info("Saving animation as GIF (Pillow writer) …")
            anim.save(str(anim_out), writer="pillow", fps=fps)
        else:
            log.info("Saving animation as MP4 (FFmpeg writer) …")
            anim.save(str(anim_out), writer="ffmpeg", fps=fps,
                      extra_args=["-vcodec", "libx264", "-crf", "22"])
        log.info("Animation saved → %s", anim_out)
    except Exception as exc:
        log.error("Animation save failed: %s", exc)
        log.info("Tip: for GIF output use a .gif extension (no ffmpeg required)")
    finally:
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LMR_surfer — Lower Mississippi River hydraulic profiler"
    )
    p.add_argument(
        "--date", default="",
        metavar="YYYY-MM-DD",
        help="Target date (default: most recent available date in the corrected stage file)",
    )
    p.add_argument(
        "--out", default="",
        metavar="PATH",
        help="Optional output file path for the figure (PNG / PDF / SVG). "
             "Defaults to models/qaqc_outputs/lmr_profile_<date>.png",
    )
    p.add_argument(
        "--list-dates", action="store_true",
        help="Print the 10 most recent available dates and exit",
    )
    p.add_argument(
        "--no-dem", action="store_true",
        help="Skip DEM sampling (faster; omits channel-bed profile and DEM-based storage)",
    )
    p.add_argument(
        "--no-show", action="store_true",
        help="Suppress the interactive plot window (write file only)",
    )
    p.add_argument(
        "--transect-km", type=float, default=TRANSECT_HALF_WIDTH_M / 1_000,
        metavar="KM",
        help=f"Half-width of DEM cross-section transects in km "
             f"(default {TRANSECT_HALF_WIDTH_M / 1000:.1f})",
    )
    p.add_argument(
        "--spike-thresh", type=float, default=5.0,
        metavar="FT",
        help="Threshold in feet for flagging hydraulically inconsistent stage spikes (default: 5.0)",
    )
    p.add_argument(
        "--adjust", nargs="+", default=[],
        metavar="ID:OFFSET",
        help="Manually apply a datum offset to specific gages (e.g. --adjust 01145:-8.5 01160:1.2)",
    )
    # ── Animation ────────────────────────────────────────────────────
    p.add_argument(
        "--start-date", default="",
        metavar="YYYY-MM-DD",
        help="Start date for animation (requires --end-date).",
    )
    p.add_argument(
        "--end-date", default="",
        metavar="YYYY-MM-DD",
        help="End date for animation (requires --start-date).",
    )
    p.add_argument(
        "--fps", type=int, default=5,
        metavar="N",
        help="Frames per second for animation output (default: 5).",
    )
    p.add_argument(
        "--anim-out", default="",
        metavar="PATH",
        help="Output path for animation (.gif or .mp4; default: auto-named in output dir).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    log.info("═" * 60)
    log.info("LMR Surfer — Lower Mississippi River Hydraulic Profiler")
    log.info("═" * 60)

    # ── Parse manual datum adjustments ────────────────────────────────
    manual_adjustments: Dict[str, float] = {}
    for token in args.adjust:
        try:
            sid, off = token.split(":", 1)
            manual_adjustments[sid.strip()] = float(off)
        except ValueError:
            log.warning("Ignoring malformed --adjust token: %s", token)

    # ── Load all data ──────────────────────────────────────────────────
    artifact      = load_artifact()
    corrected     = load_corrected_stage()
    all_discharge = load_all_discharge_series()   # dict: usgs_id → Series + "primary"
    discharge     = all_discharge.get("primary", pd.Series(dtype=float))
    sta_miles     = get_station_miles(artifact)

    # ── List dates mode ────────────────────────────────────────────────
    if args.list_dates:
        recent = corrected.index.sort_values(ascending=False)[:10]
        print("10 most recent available dates in corrected stage file:")
        for d in recent:
            print(f"  {d.date()}")
        return

    # ── River centreline ───────────────────────────────────────────────
    try:
        centreline = load_lmr_centreline()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    # ── DEM profiler ───────────────────────────────────────────────────
    if args.no_dem:
        log.info("--no-dem: DEM processing skipped")
        dem = DEMProfiler(Path("__nonexistent_dem__.tif"))
    else:
        if not HAS_RASTERIO:
            log.warning(
                "rasterio not installed — DEM channel-bed profile unavailable.\n"
                "  Install with:  pip install rasterio\n"
                "  Using geometric fallback for storage."
            )
        dem = DEMProfiler(DEM_PATH)

    # ── Hysteresis envelopes ───────────────────────────────────────────
    log.info("Computing historical stage-discharge envelopes …")
    hysteresis = HysteresisAnalyzer(corrected, discharge, sta_miles)
    _ = hysteresis.envelopes   # trigger computation

    # ── StorageCalculator (shared) ─────────────────────────────────────
    storage_calc = StorageCalculator(dem)

    # ═══════════════════════════════════════════════════════════════════
    # ANIMATION MODE
    # ═══════════════════════════════════════════════════════════════════
    if args.start_date and args.end_date:
        try:
            dt_start = pd.Timestamp(args.start_date)
            dt_end   = pd.Timestamp(args.end_date)
        except Exception as exc:
            log.error("Invalid date range: %s", exc)
            sys.exit(1)

        all_dates = pd.date_range(dt_start, dt_end, freq="D")
        # Keep only dates that exist in the corrected stage index
        avail_dates = [d for d in all_dates if d in corrected.index]
        if not avail_dates:
            log.error("No corrected stage data between %s and %s — aborting",
                      dt_start.date(), dt_end.date())
            sys.exit(1)
        log.info("Animation: %d frames from %s to %s",
                 len(avail_dates), avail_dates[0].date(), avail_dates[-1].date())

        anim_path = (
            Path(args.anim_out)
            if args.anim_out
            else OUTPUT_DIR / (
                f"lmr_animation_{dt_start.strftime('%Y%m%d')}"
                f"_{dt_end.strftime('%Y%m%d')}.gif"
            )
        )
        animate_profile(
            dates              = avail_dates,
            corrected          = corrected,
            all_discharge      = all_discharge,
            sta_miles          = sta_miles,
            centreline         = centreline,
            dem                = dem,
            hysteresis         = hysteresis,
            spike_thresh       = args.spike_thresh,
            transect_km        = args.transect_km,
            manual_adjustments = manual_adjustments,
            anim_out           = anim_path,
            fps                = args.fps,
        )
        return

    # ═══════════════════════════════════════════════════════════════════
    # SINGLE-FRAME MODE
    # ═══════════════════════════════════════════════════════════════════
    target_date = (
        pd.Timestamp(args.date) if args.date else corrected.index.max()
    )
    if not args.date:
        log.info("No --date specified — using most recent: %s", target_date.date())

    # Reference discharge on target date
    q_cfs = 0.0
    if not discharge.empty:
        dt_idx = int(np.abs((discharge.index - target_date).total_seconds()).argmin())
        q_cfs  = float(discharge.iloc[dt_idx])
        log.info("Reference discharge on %s: %.0f cfs", target_date.date(), q_cfs)

    # Per-station envelope stats for the current discharge bin
    envelope_at_q: Dict[str, Dict] = {
        sid: env
        for sid in sta_miles
        for env in [hysteresis.get_envelope(sid, q_cfs)]
        if env is not None
    }

    # Gap-fill staleness warnings
    raw_stage = load_raw_stage()
    if raw_stage is not None:
        for sid in sorted(sta_miles, key=lambda s: sta_miles[s], reverse=True):
            col = f"stage_{sid}"
            if col not in raw_stage.columns:
                continue
            series_to_date = raw_stage[col].loc[:target_date]
            consecutive_nan = int(
                series_to_date.iloc[::-1].isna().cumprod().sum()
            )
            if consecutive_nan > 14:
                log.warning(
                    "STALE gap-fill at %s: %d consecutive days without raw "
                    "observation (last raw obs: %s) — model predictions may be unreliable",
                    sid,
                    consecutive_nan,
                    series_to_date.last_valid_index().date()
                    if series_to_date.last_valid_index() is not None
                    else "never",
                )

    # Build hydraulic profile
    profiler = HydraulicProfiler(corrected, sta_miles, centreline, dem)
    log.info("Building longitudinal hydraulic profile for %s …", target_date.date())
    try:
        profile_df = profiler.build_profile(
            target_date,
            spike_threshold_ft  = args.spike_thresh,
            envelope_at_q       = envelope_at_q,
            manual_adjustments  = manual_adjustments,
        )
    except ValueError as exc:
        log.error("Profile build failed: %s", exc)
        sys.exit(1)

    stages = profiler.get_stage_at_date(target_date)
    log.info(
        "Stage readings at %d gages:\n  %s",
        len(stages),
        "  ".join(
            f"{sid}={v:.1f}ft"
            for sid, v in sorted(stages.items(),
                                 key=lambda kv: sta_miles.get(kv[0], 0),
                                 reverse=True)
        ),
    )

    # In-channel storage
    log.info("Computing in-channel storage volumes …")
    half_width_m = args.transect_km * 1_000
    storage_df   = storage_calc.compute(profile_df, half_width_m=half_width_m,
                                        gage_only=True)
    total_maf = float(storage_df["cum_storage_af"].iloc[-1]) / 1e6
    source = "DEM cross-sections" if dem.available else "geometric fallback"
    log.info("Total estimated in-channel storage: %.3f million acre-feet (%s)",
             total_maf, source)

    # Historical P05/P50/P95 envelope areas
    log.info("Computing historical cross-section envelope areas at Q=%.0f cfs …", q_cfs)
    env_areas = storage_calc.compute_stage_envelope_areas(
        storage_df, hysteresis, q_cfs, half_width_m=half_width_m
    )
    for col, arr in env_areas.items():
        storage_df[col] = arr

    n_env = sum(1 for sid in stages if hysteresis.get_envelope(sid, q_cfs))
    log.info("Hysteresis envelopes available for %d / %d gages", n_env, len(stages))

    # Limb classification log
    for sid, s_ft in sorted(stages.items(),
                             key=lambda kv: sta_miles.get(kv[0], 0), reverse=True):
        limb = hysteresis.classify_limb(sid, s_ft, q_cfs)
        env  = hysteresis.get_envelope(sid, q_cfs)
        p50  = f"{env['p50']:.1f}" if env else "—"
        log.info(
            "  %-8s  stage=%5.1f ft  limb=%-8s  median(Q-bin)=%s ft",
            sid, s_ft, limb, p50,
        )

    # Spatial discharge profile (illustrates unsteady flow)
    log.info("Computing spatially-varying Q profile from %d USGS main-stem gauges …",
             sum(1 for sid in USGS_Q_MAINSTEM if sid in all_discharge))
    spatial_q = compute_spatial_q_profile(
        profile_df["river_mile"].values, all_discharge, target_date
    )

    # Historical stage envelope interpolated along full profile
    stage_env = compute_stage_envelope_profile(profile_df, hysteresis, q_cfs)

    # Figure output path
    fig_out = (
        Path(args.out)
        if args.out
        else OUTPUT_DIR / f"lmr_profile_{target_date.strftime('%Y%m%d')}.png"
    )
    log.info("Writing figure → %s", fig_out)

    plot_hydraulic_profile(
        profile       = profile_df,
        storage       = storage_df,
        stages        = stages,
        station_miles = sta_miles,
        hysteresis    = hysteresis,
        q_cfs         = q_cfs,
        target_date   = target_date,
        flagged_gages = getattr(profiler, "flagged_gages", {}),
        stage_env     = stage_env,
        spatial_q     = spatial_q,
        fig_out       = fig_out,
        show          = not args.no_show,
    )


if __name__ == "__main__":
    main()
