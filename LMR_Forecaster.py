#!/usr/bin/env python3
"""
LMR_Forecaster.py – Lower Mississippi River Stage / Discharge Forecaster
========================================================================

Predicts river stage at any river mile on the Lower Mississippi River (LMR)
for up to 28 days from the current date.

Data sources
------------
* **USGS NWIS** – river stage (parameter 00065) and discharge (00060) from
  gauges on the Mississippi mainstem, Ohio, Missouri, Arkansas, White, Yazoo,
  and Red rivers.
* **NOAA CO-OPS** – water-level observations from tide stations at Grand Isle,
  LA (8761724) and Pensacola, FL (8729840), which bracket the Gulf base-level
  influence on the LMR delta.  Grand Isle is closer to the river but subject
  to land subsidence; Pensacola provides a geoidally stable reference.

Machine-learning approach
-------------------------
* XGBoost gradient-boosted regression trained per gauge.
* Feature set: lagged stage/discharge, rates of change, rolling statistics,
  upstream-neighbour values with travel-time offsets, tidal base level, and
  sinusoidal day-of-year encoding for seasonality.
* Exponential-decay sample weighting so that recent years dominate while still
  learning from the full historical record—handles slow drift caused by river
  engineering, subsidence, land-use change, and datum shifts.
* Hierarchical recursive multi-step forecasting: tide gauges first, then
  tributaries, then mainstem from upstream to downstream.
* Anomaly detection compares each gauge against its network-predicted value,
  flagging sudden datum drift or equipment failure.

Usage examples
--------------
    # Forecast at Vicksburg (RM 437) for 28 days
    python LMR_Forecaster.py --mile 437

    # Forecast at a lat/lon for 14 days
    python LMR_Forecaster.py --lat 30.44 --lon -91.19 --days 14

    # Retrain on 10 years of history
    python LMR_Forecaster.py --train --years 10

    # Check gauges for anomalies
    python LMR_Forecaster.py --check-anomalies

Requirements
------------
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("LMR_Forecaster")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
CACHE_DIR = Path("./cache")
MODEL_DIR = Path("./models")
HISTORICAL_YEARS_DEFAULT = 10
RECENT_DAYS = 60
FORECAST_HORIZON = 28
USGS_BASE = "https://waterservices.usgs.gov/nwis"
NOAA_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
REQUEST_TIMEOUT = 120  # seconds
MAX_WORKERS = 4  # parallel HTTP threads

# XGBoost hyper-parameters (tuned for daily river stage regression)
XGB_PARAMS: dict = dict(
    objective="reg:squarederror",
    max_depth=6,
    learning_rate=0.05,
    n_estimators=800,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
)
XGB_EARLY_STOP = 50

# Exponential-decay half-life (days).  Data from one half-life ago receives
# half the weight of today's data.  365 days is a good default that lets the
# model gradually forget obsolete relationships while retaining seasonal
# patterns across at least a couple of complete cycles.
DECAY_HALF_LIFE_DAYS = 365

# Anomaly detection thresholds
ANOMALY_ZSCORE_THRESHOLD = 3.0
DRIFT_WINDOW_DAYS = 14
DRIFT_SHIFT_THRESHOLD = 2.0  # σ of rolling-mean shift

# Approximate flow speed on the LMR in miles per day (used for travel-time
# lag estimation between gauges).  Varies with discharge; 100 mi/day is a
# reasonable moderate-flow average.
FLOW_SPEED_MI_PER_DAY = 100.0


# ═══════════════════════════════════════════════════════════════════════════
# Gauge station definition
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class GaugeStation:
    """Metadata for a single gauge in the monitoring network."""

    station_id: str  # USGS site number or NOAA station ID
    name: str
    source: str  # "usgs" | "noaa"
    river: str  # e.g. "mississippi", "ohio", "gulf"
    river_mile: float  # approximate LMR river mile (Head-of-Passes = 0)
    latitude: float
    longitude: float
    param_codes: List[str] = field(default_factory=lambda: ["00065", "00060"])
    is_tidal: bool = False
    # Forecast order: lower number → predicted first.
    # 0 = tidal / exogenous, 1 = tributary, 2 = upper mainstem, 3 = lower mainstem
    forecast_tier: int = 3

    @property
    def cache_key(self) -> str:
        return f"{self.source}_{self.station_id}"


# ---------------------------------------------------------------------------
# Gauge network – the stations used by the forecaster
# ---------------------------------------------------------------------------
def _build_gauge_network() -> List[GaugeStation]:
    """Return the default gauge network for the LMR forecaster."""
    stations: List[GaugeStation] = [
        # ── Mississippi mainstem (upstream → downstream) ──────────────
        GaugeStation(
            "05587450", "Mississippi at Grafton, IL", "usgs", "mississippi",
            river_mile=1070, latitude=38.97, longitude=-90.34,
            forecast_tier=2,
        ),
        GaugeStation(
            "07010000", "Mississippi at St. Louis, MO", "usgs", "mississippi",
            river_mile=1040, latitude=38.63, longitude=-90.18,
            forecast_tier=2,
        ),
        GaugeStation(
            "07020500", "Mississippi at Chester, IL", "usgs", "mississippi",
            river_mile=990, latitude=37.91, longitude=-89.83,
            forecast_tier=2,
        ),
        GaugeStation(
            "07022000", "Mississippi at Thebes, IL", "usgs", "mississippi",
            river_mile=960, latitude=37.22, longitude=-89.47,
            forecast_tier=2,
        ),
        # ── Lower Mississippi ─────────────────────────────────────────
        GaugeStation(
            "07032000", "Mississippi at Memphis, TN", "usgs", "mississippi_lower",
            river_mile=736, latitude=35.13, longitude=-90.07,
            forecast_tier=3,
        ),
        GaugeStation(
            "07289000", "Mississippi at Vicksburg, MS", "usgs", "mississippi_lower",
            river_mile=437, latitude=32.31, longitude=-90.91,
            forecast_tier=3,
        ),
        GaugeStation(
            "07295100", "Mississippi at Tarbert Landing, MS", "usgs", "mississippi_lower",
            river_mile=306, latitude=31.00, longitude=-91.62,
            forecast_tier=3,
        ),
        GaugeStation(
            "07381600", "Lower Atchafalaya at Morgan City, LA", "usgs", "atchafalaya",
            river_mile=0, latitude=29.69, longitude=-91.21,
            forecast_tier=3,
        ),
        GaugeStation(
            "07374000", "Mississippi at Baton Rouge, LA", "usgs", "mississippi_lower",
            river_mile=228, latitude=30.44, longitude=-91.19,
            forecast_tier=3,
        ),
        GaugeStation(
            "07374525", "Mississippi at Belle Chasse, LA", "usgs", "mississippi_lower",
            river_mile=76, latitude=29.85, longitude=-89.99,
            forecast_tier=3,
        ),
        # ── Major tributaries ─────────────────────────────────────────
        GaugeStation(
            "06934500", "Missouri at Hermann, MO", "usgs", "missouri",
            river_mile=1060, latitude=38.71, longitude=-91.44,
            forecast_tier=1,
        ),
        GaugeStation(
            "03611500", "Ohio at Metropolis, IL", "usgs", "ohio",
            river_mile=953, latitude=37.15, longitude=-88.75,
            forecast_tier=1,
        ),
        GaugeStation(
            "03612600", "Ohio at Olmsted, IL", "usgs", "ohio",
            river_mile=953, latitude=37.18, longitude=-89.08,
            forecast_tier=1,
        ),
        GaugeStation(
            "07077380", "White at Clarendon, AR", "usgs", "white",
            river_mile=600, latitude=34.69, longitude=-91.31,
            forecast_tier=1,
        ),
        GaugeStation(
            "07263620", "Arkansas at David D. Terry L&D, AR", "usgs", "arkansas",
            river_mile=580, latitude=34.68, longitude=-92.09,
            forecast_tier=1,
        ),
        GaugeStation(
            "07265450", "Mississippi at Arkansas City, AR", "usgs", "mississippi_lower",
            river_mile=554, latitude=33.56, longitude=-91.24,
            forecast_tier=3,
        ),
        GaugeStation(
            "07355500", "Red at Alexandria, LA", "usgs", "red",
            river_mile=310, latitude=31.31, longitude=-92.45,
            forecast_tier=1,
        ),
        GaugeStation(
            "07288955", "Yazoo below Steele Bayou, MS", "usgs", "yazoo",
            river_mile=440, latitude=32.60, longitude=-90.94,
            forecast_tier=1,
        ),
        # ── NOAA tide gauges (Gulf base level) ────────────────────────
        GaugeStation(
            "8761724", "Grand Isle, LA", "noaa", "gulf",
            river_mile=-50, latitude=29.26, longitude=-89.96,
            param_codes=[], is_tidal=True, forecast_tier=0,
        ),
        GaugeStation(
            "8729840", "Pensacola, FL", "noaa", "gulf",
            river_mile=-200, latitude=30.40, longitude=-87.21,
            param_codes=[], is_tidal=True, forecast_tier=0,
        ),
    ]
    return stations


GAUGE_NETWORK: List[GaugeStation] = _build_gauge_network()


def get_gauge(station_id: str) -> Optional[GaugeStation]:
    """Look up a gauge by station_id."""
    for g in GAUGE_NETWORK:
        if g.station_id == station_id:
            return g
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Data fetching – USGS NWIS
# ═══════════════════════════════════════════════════════════════════════════
class USGSClient:
    """Thin wrapper around the USGS NWIS daily-values and instantaneous-
    values REST services."""

    def fetch_daily(
        self,
        site_id: str,
        start_date: str,
        end_date: str,
        param_codes: List[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch daily-value records and return a DataFrame indexed by date.

        Columns are named ``<param_code>`` (e.g. ``00065`` for stage in ft,
        ``00060`` for discharge in cfs).
        """
        if param_codes is None:
            param_codes = ["00065", "00060"]
        params = {
            "format": "json",
            "sites": site_id,
            "startDT": start_date,
            "endDT": end_date,
            "parameterCd": ",".join(param_codes),
            "siteStatus": "all",
        }
        url = f"{USGS_BASE}/dv/"
        return self._parse_nwis(url, params, site_id)

    def fetch_iv(
        self,
        site_id: str,
        start_date: str,
        end_date: str,
        param_codes: List[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch instantaneous values, resample to daily, and return a
        DataFrame indexed by date."""
        if param_codes is None:
            param_codes = ["00065", "00060"]
        params = {
            "format": "json",
            "sites": site_id,
            "startDT": start_date,
            "endDT": end_date,
            "parameterCd": ",".join(param_codes),
            "siteStatus": "all",
        }
        url = f"{USGS_BASE}/iv/"
        df = self._parse_nwis(url, params, site_id)
        if df.empty:
            return df
        # Resample instantaneous → daily mean
        df = df.resample("D").mean()
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_nwis(url: str, params: dict, site_id: str) -> pd.DataFrame:
        """Issue the HTTP request and parse the NWIS JSON response."""
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("USGS request failed for site %s: %s", site_id, exc)
            return pd.DataFrame()

        try:
            data = resp.json()
        except json.JSONDecodeError:
            log.warning("USGS returned non-JSON for site %s", site_id)
            return pd.DataFrame()

        ts_list = data.get("value", {}).get("timeSeries", [])
        if not ts_list:
            log.warning("No time series returned for USGS site %s", site_id)
            return pd.DataFrame()

        frames: Dict[str, pd.Series] = {}
        for ts in ts_list:
            var_code = ts["variable"]["variableCode"][0]["value"]
            values = ts.get("values", [{}])[0].get("value", [])
            if not values:
                continue
            dates, vals = [], []
            for v in values:
                raw = v.get("value")
                if raw is None or float(raw) <= -999999:
                    continue
                dates.append(pd.Timestamp(v["dateTime"]))
                vals.append(float(raw))
            if dates:
                s = pd.Series(vals, index=pd.DatetimeIndex(dates), name=var_code)
                # In case there are duplicate timestamps (multi-method), keep last
                s = s[~s.index.duplicated(keep="last")]
                frames[var_code] = s

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index.name = "date"
        df = df.sort_index()
        return df


# ═══════════════════════════════════════════════════════════════════════════
# Data fetching – NOAA CO-OPS
# ═══════════════════════════════════════════════════════════════════════════
class NOAAClient:
    """Thin wrapper around the NOAA CO-OPS API for tide-gauge water levels."""

    MAX_RANGE_DAYS = 30  # NOAA water_level allows max 31 days per request

    def fetch_water_level(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Return daily-mean water level (ft, STND datum) for *station_id*.

        Uses the 6-minute ``water_level`` product (available for all coastal
        stations) and resamples to daily means.
        """
        sd = pd.Timestamp(start_date)
        ed = pd.Timestamp(end_date)
        all_frames: List[pd.DataFrame] = []

        # NOAA limits water_level to 31 days; chunk into 30-day blocks
        chunk_start = sd
        while chunk_start < ed:
            chunk_end = min(chunk_start + timedelta(days=self.MAX_RANGE_DAYS), ed)
            df_chunk = self._fetch_chunk(
                station_id,
                chunk_start.strftime("%Y%m%d"),
                chunk_end.strftime("%Y%m%d"),
            )
            if not df_chunk.empty:
                all_frames.append(df_chunk)
            chunk_start = chunk_end + timedelta(days=1)

        if not all_frames:
            return pd.DataFrame()
        df = pd.concat(all_frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        # Resample 6-minute data to daily mean
        df = df.resample("D").mean()
        df = df.dropna(how="all")
        return df

    @staticmethod
    def _fetch_chunk(station_id: str, begin: str, end: str) -> pd.DataFrame:
        params = {
            "begin_date": begin,
            "end_date": end,
            "station": station_id,
            "product": "water_level",
            "datum": "STND",
            "units": "english",
            "time_zone": "gmt",
            "application": "LMR_Forecaster",
            "format": "json",
        }
        try:
            resp = requests.get(NOAA_BASE, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            log.warning("NOAA request failed for %s: %s", station_id, exc)
            return pd.DataFrame()

        records = payload.get("data", [])
        if not records:
            # NOAA returns an error dict when data is unavailable
            log.debug("No NOAA data for station %s (%s–%s)", station_id, begin, end)
            return pd.DataFrame()

        dates, vals = [], []
        for r in records:
            try:
                v = float(r["v"])
                dates.append(pd.Timestamp(r["t"]))
                vals.append(v)
            except (ValueError, KeyError):
                continue
        if not dates:
            return pd.DataFrame()

        df = pd.DataFrame({"water_level": vals}, index=pd.DatetimeIndex(dates, name="date"))
        return df


# ═══════════════════════════════════════════════════════════════════════════
# Data caching
# ═══════════════════════════════════════════════════════════════════════════
class DataCache:
    """Simple pickle-based cache for gauge time-series data."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key}_{safe}.pkl"

    def load(self, key: str) -> Optional[pd.DataFrame]:
        p = self._path(key)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                log.warning("Corrupt cache for %s – refetching", key)
        return None

    def save(self, key: str, df: pd.DataFrame) -> None:
        with open(self._path(key), "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    def clear(self) -> None:
        for p in self.cache_dir.glob("*.pkl"):
            p.unlink()


# ═══════════════════════════════════════════════════════════════════════════
# Data manager – orchestrates fetching + caching for the whole network
# ═══════════════════════════════════════════════════════════════════════════
class DataManager:
    """Fetch, cache, and align data for the full gauge network."""

    def __init__(
        self,
        network: List[GaugeStation] | None = None,
        cache_dir: Path = CACHE_DIR,
    ):
        self.network = network or GAUGE_NETWORK
        self.cache = DataCache(cache_dir)
        self.usgs = USGSClient()
        self.noaa = NOAAClient()

    # ------------------------------------------------------------------
    def fetch_gauge(
        self,
        gauge: GaugeStation,
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch data for a single gauge, using cache if available."""
        key = f"{gauge.cache_key}_{start}_{end}"
        if use_cache:
            cached = self.cache.load(key)
            if cached is not None:
                log.debug("Cache hit: %s", gauge.name)
                return cached

        log.info("Fetching %s (%s) %s → %s …", gauge.name, gauge.station_id, start, end)
        if gauge.source == "usgs":
            df = self.usgs.fetch_daily(gauge.station_id, start, end, gauge.param_codes)
        elif gauge.source == "noaa":
            df = self.noaa.fetch_water_level(gauge.station_id, start, end)
        else:
            log.warning("Unknown source '%s' for gauge %s", gauge.source, gauge.name)
            df = pd.DataFrame()

        if not df.empty:
            self.cache.save(key, df)
        return df

    # ------------------------------------------------------------------
    def fetch_all(
        self,
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for every gauge in the network (parallel).

        Returns ``{station_id: DataFrame}``.
        """
        results: Dict[str, pd.DataFrame] = {}

        def _worker(g: GaugeStation) -> Tuple[str, pd.DataFrame]:
            return g.station_id, self.fetch_gauge(g, start, end, use_cache)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_worker, g): g for g in self.network}
            for f in as_completed(futures):
                g = futures[f]
                try:
                    sid, df = f.result()
                    results[sid] = df
                except Exception as exc:
                    log.warning("Failed to fetch %s: %s", g.name, exc)

        n_ok = sum(1 for d in results.values() if not d.empty)
        log.info(
            "Fetched data for %d / %d gauges (%d with data)",
            len(results), len(self.network), n_ok,
        )
        return results

    # ------------------------------------------------------------------
    def build_aligned_daily(
        self,
        gauge_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge per-gauge DataFrames into a single wide DataFrame aligned
        on a common daily date index.

        Column names are ``{station_id}_{variable}`` – e.g. ``07010000_00065``
        for the stage at St. Louis.  NOAA water-level columns are named
        ``{station_id}_wl``.
        """
        frames: List[pd.Series] = []
        for sid, df in gauge_data.items():
            if df.empty:
                continue
            # Ensure daily frequency
            df = df.resample("D").mean()
            for col in df.columns:
                suffix = col if col != "water_level" else "wl"
                s = df[col].rename(f"{sid}_{suffix}")
                frames.append(s)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, axis=1).sort_index()
        # Forward-fill short gaps (≤3 days), then leave remaining as NaN
        merged = merged.ffill(limit=3)
        return merged


# ═══════════════════════════════════════════════════════════════════════════
# Feature engineering
# ═══════════════════════════════════════════════════════════════════════════
class FeatureEngineer:
    """Build ML features from a wide aligned DataFrame."""

    # Lag windows (days)
    LAGS = [1, 2, 3, 5, 7, 14, 21, 28]
    # Rolling-statistics windows
    ROLL_WINDOWS = [7, 14, 28]

    def __init__(self, network: List[GaugeStation] | None = None):
        self.network = network or GAUGE_NETWORK
        self._gauge_map: Dict[str, GaugeStation] = {g.station_id: g for g in self.network}

    # ------------------------------------------------------------------
    def build_training_set(
        self,
        aligned: pd.DataFrame,
        target_gauge_id: str,
        target_var: str = "00065",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Create feature matrix **X** and target vector **y** for one gauge.

        ``target_var`` is the USGS parameter code for the prediction target
        (``"00065"`` = stage, ``"00060"`` = discharge).  For NOAA gauges the
        target variable is ``"wl"`` (water level).
        """
        target_col = f"{target_gauge_id}_{target_var}"
        if target_col not in aligned.columns:
            # Try water_level for NOAA gauges
            target_col = f"{target_gauge_id}_wl"
        if target_col not in aligned.columns:
            raise KeyError(f"Target column {target_col} not found in aligned data")

        features: Dict[str, pd.Series] = {}

        # --- 1. Target-gauge lagged values & rate of change ---------------
        self._add_lag_features(aligned, target_col, features, prefix="tgt")
        self._add_rate_features(aligned, target_col, features, prefix="tgt")
        self._add_roll_features(aligned, target_col, features, prefix="tgt")

        # --- 2. Neighbour-gauge features (with travel-time lag) -----------
        target_gauge = self._gauge_map.get(target_gauge_id)
        for sid, gauge in self._gauge_map.items():
            if sid == target_gauge_id:
                continue
            # Pick the best variable column for this neighbour
            for var in ["00065", "wl", "00060"]:
                ncol = f"{sid}_{var}"
                if ncol in aligned.columns:
                    break
            else:
                continue

            # Estimate travel-time offset (days) between neighbour and target
            if target_gauge is not None:
                tt = self._travel_time(gauge, target_gauge)
            else:
                tt = 0
            tt_lag = max(1, int(round(tt)))

            prefix = f"n{sid}"
            # Add lagged values with travel-time shift
            for lag in [tt_lag, tt_lag + 1, tt_lag + 3, tt_lag + 7]:
                key = f"{prefix}_lag{lag}"
                features[key] = aligned[ncol].shift(lag)

            # Add rolling mean of neighbour (7-day) for smoothness
            features[f"{prefix}_rm7"] = aligned[ncol].rolling(7, min_periods=1).mean().shift(tt_lag)

        # --- 3. Seasonal features ─────────────────────────────────────────
        doy = aligned.index.dayofyear.values
        features["sin_doy"] = pd.Series(np.sin(2 * np.pi * doy / 365.25), index=aligned.index)
        features["cos_doy"] = pd.Series(np.cos(2 * np.pi * doy / 365.25), index=aligned.index)
        features["sin_doy2"] = pd.Series(np.sin(4 * np.pi * doy / 365.25), index=aligned.index)
        features["cos_doy2"] = pd.Series(np.cos(4 * np.pi * doy / 365.25), index=aligned.index)
        features["month"] = pd.Series(aligned.index.month, index=aligned.index, dtype=float)

        # Build X
        X = pd.DataFrame(features, index=aligned.index)

        # y = next-day target value (1-step-ahead)
        y = aligned[target_col].shift(-1)

        # Drop rows with NaN target or insufficient features
        mask = y.notna() & (X.notna().sum(axis=1) > X.shape[1] * 0.3)
        X = X.loc[mask]
        y = y.loc[mask]

        # Fill remaining NaN features with column median
        X = X.fillna(X.median())

        return X, y

    # ------------------------------------------------------------------
    def build_forecast_row(
        self,
        aligned: pd.DataFrame,
        target_gauge_id: str,
        target_var: str = "00065",
    ) -> pd.DataFrame:
        """Build a single-row feature vector from the latest available data
        (i.e. the last row of *aligned*), suitable for predicting the next
        day's value.
        """
        X, _ = self.build_training_set(aligned, target_gauge_id, target_var)
        if X.empty:
            return pd.DataFrame()
        return X.iloc[[-1]]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _add_lag_features(
        df: pd.DataFrame, col: str, out: Dict[str, pd.Series], prefix: str
    ) -> None:
        for lag in FeatureEngineer.LAGS:
            out[f"{prefix}_lag{lag}"] = df[col].shift(lag)

    @staticmethod
    def _add_rate_features(
        df: pd.DataFrame, col: str, out: Dict[str, pd.Series], prefix: str
    ) -> None:
        for w in [1, 3, 7]:
            out[f"{prefix}_delta{w}"] = df[col].diff(w)

    @staticmethod
    def _add_roll_features(
        df: pd.DataFrame, col: str, out: Dict[str, pd.Series], prefix: str
    ) -> None:
        for w in FeatureEngineer.ROLL_WINDOWS:
            r = df[col].rolling(w, min_periods=1)
            out[f"{prefix}_rmean{w}"] = r.mean()
            out[f"{prefix}_rstd{w}"] = r.std()

    @staticmethod
    def _travel_time(source: GaugeStation, target: GaugeStation) -> float:
        """Approximate travel time in days between two gauges."""
        dist = abs(source.river_mile - target.river_mile)
        if dist == 0:
            return 1.0
        return dist / FLOW_SPEED_MI_PER_DAY


# ═══════════════════════════════════════════════════════════════════════════
# Per-gauge XGBoost model
# ═══════════════════════════════════════════════════════════════════════════
class StageModel:
    """Wraps an XGBoost regressor for a single gauge station."""

    def __init__(self, station_id: str, params: dict | None = None):
        self.station_id = station_id
        self.params = params or dict(XGB_PARAMS)
        self.model: Any = None  # xgb.XGBRegressor after training
        self.feature_names: List[str] = []
        self.train_mae: float = np.nan
        self.val_mae: float = np.nan

    # ------------------------------------------------------------------
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """Train the model with time-series-aware validation."""
        import xgboost as xgb  # deferred import

        self.feature_names = list(X.columns)

        # Time-series split: last 20 % of data as validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        if sample_weights is not None:
            sw_train = sample_weights[:split_idx]
        else:
            sw_train = None

        params = dict(self.params)
        n_est = params.pop("n_estimators", 800)
        early = XGB_EARLY_STOP

        reg = xgb.XGBRegressor(n_estimators=n_est, early_stopping_rounds=early, **params)

        eval_set = [(X_val.values, y_val.values)]
        reg.fit(
            X_train.values,
            y_train.values,
            sample_weight=sw_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.model = reg
        self.train_mae = mean_absolute_error(y_train, reg.predict(X_train.values))
        self.val_mae = mean_absolute_error(y_val, reg.predict(X_val.values))
        log.info(
            "  %s – train MAE=%.3f  val MAE=%.3f  (n_train=%d, n_val=%d, n_feat=%d)",
            self.station_id, self.train_mae, self.val_mae,
            len(X_train), len(X_val), X_train.shape[1],
        )

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        # Ensure feature alignment
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0.0)
        return self.model.predict(X_aligned.values)

    # ------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"model_{self.station_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, directory: Path, station_id: str) -> "StageModel":
        path = directory / f"model_{station_id}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    def feature_importance(self, top_n: int = 15) -> pd.Series:
        """Return top-N feature importances."""
        if self.model is None:
            return pd.Series(dtype=float)
        imp = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)
        return imp.head(top_n)


# ═══════════════════════════════════════════════════════════════════════════
# Network forecaster – hierarchical multi-step predictions
# ═══════════════════════════════════════════════════════════════════════════
class NetworkForecaster:
    """Trains one StageModel per gauge and generates recursive multi-step
    forecasts respecting the upstream → downstream hierarchy."""

    def __init__(
        self,
        network: List[GaugeStation] | None = None,
        model_dir: Path = MODEL_DIR,
    ):
        self.network = network or GAUGE_NETWORK
        self.model_dir = model_dir
        self.models: Dict[str, StageModel] = {}
        self.feat_eng = FeatureEngineer(self.network)
        # Per-gauge validation errors by forecast horizon (for confidence bounds)
        self.horizon_errors: Dict[str, Dict[int, float]] = {}

    # ------------------------------------------------------------------
    def train_all(
        self,
        aligned: pd.DataFrame,
        half_life_days: int = DECAY_HALF_LIFE_DAYS,
    ) -> None:
        """Train a model for each gauge in the network."""
        log.info("Training models for %d gauges …", len(self.network))

        # Track which variable each model predicts
        self.model_target_vars: Dict[str, str] = {}

        for gauge in sorted(self.network, key=lambda g: g.forecast_tier):
            sid = gauge.station_id
            tvar = self._best_target_var(gauge, aligned)

            # Check if target column exists
            tcol = f"{sid}_{tvar}"
            if tcol not in aligned.columns:
                log.warning("  Skipping %s – no target column '%s'", gauge.name, tcol)
                continue

            try:
                X, y = self.feat_eng.build_training_set(aligned, sid, tvar)
            except KeyError as exc:
                log.warning("  Skipping %s – %s", gauge.name, exc)
                continue

            if len(X) < 60:
                log.warning("  Skipping %s – only %d samples", gauge.name, len(X))
                continue

            # Exponential-decay sample weights
            weights = self._decay_weights(X.index, half_life_days)

            model = StageModel(sid)
            model.train(X, y, weights)
            self.models[sid] = model
            self.model_target_vars[sid] = tvar
            model.save(self.model_dir)

            # Estimate horizon-dependent errors via rolling validation
            self._estimate_horizon_errors(aligned, gauge, tvar)

        log.info("Training complete – %d models ready.", len(self.models))

    # ------------------------------------------------------------------
    def load_models(self) -> int:
        """Load previously saved models from disk. Returns count loaded."""
        loaded = 0
        for gauge in self.network:
            path = self.model_dir / f"model_{gauge.station_id}.pkl"
            if path.exists():
                try:
                    self.models[gauge.station_id] = StageModel.load(
                        self.model_dir, gauge.station_id
                    )
                    loaded += 1
                except Exception as exc:
                    log.warning("Failed to load model for %s: %s", gauge.name, exc)
        log.info("Loaded %d pre-trained models.", loaded)
        return loaded

    # ------------------------------------------------------------------
    def forecast(
        self,
        aligned: pd.DataFrame,
        days: int = FORECAST_HORIZON,
    ) -> pd.DataFrame:
        """Generate *days*-ahead forecasts for every gauge with a trained
        model.

        Returns a DataFrame indexed by forecast date with columns
        ``{station_id}_pred`` for each gauge.
        """
        if not self.models:
            raise RuntimeError("No trained models – call train_all() first")

        last_date = aligned.index[-1]
        forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=days, freq="D")

        # Working copy – we progressively append predicted rows
        working = aligned.copy()

        results: Dict[str, List[float]] = {sid: [] for sid in self.models}

        # Gauge ordering: respect forecast_tier (tide → tribut. → mainstem)
        ordered = sorted(
            [g for g in self.network if g.station_id in self.models],
            key=lambda g: (g.forecast_tier, -g.river_mile),
        )

        for step, fdate in enumerate(forecast_dates, 1):
            # Predict each gauge in hierarchy order
            new_row: Dict[str, float] = {}

            for gauge in ordered:
                sid = gauge.station_id
                tvar = self.model_target_vars.get(sid, self._target_var(gauge))
                tcol = f"{sid}_{tvar}"
                model = self.models.get(sid)
                if model is None:
                    continue

                try:
                    row = self.feat_eng.build_forecast_row(working, sid, tvar)
                except Exception:
                    results[sid].append(np.nan)
                    continue

                if row.empty:
                    results[sid].append(np.nan)
                    continue

                pred = float(model.predict(row)[0])
                results[sid].append(pred)
                new_row[tcol] = pred

            # Append predicted row to working df for next step's features
            if new_row:
                new_series = pd.Series(new_row, name=fdate)
                # Carry forward other columns from last known row
                last_vals = working.iloc[-1].copy()
                for k, v in new_row.items():
                    last_vals[k] = v
                last_vals.name = fdate
                working = pd.concat([working, last_vals.to_frame().T])

        # Build output DataFrame
        out = pd.DataFrame(results, index=forecast_dates)
        out.columns = [f"{c}_pred" for c in out.columns]

        # Add confidence bounds (±MAE scaled by sqrt(horizon))
        for gauge in ordered:
            sid = gauge.station_id
            pcol = f"{sid}_pred"
            if pcol not in out.columns:
                continue
            # Simple uncertainty scaling: error grows with √horizon
            base_mae = self.models[sid].val_mae if sid in self.models else 1.0
            for i, fdate in enumerate(forecast_dates):
                horizon = i + 1
                err = base_mae * np.sqrt(horizon)
                out.loc[fdate, f"{sid}_lo"] = out.loc[fdate, pcol] - err
                out.loc[fdate, f"{sid}_hi"] = out.loc[fdate, pcol] + err

        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _target_var(gauge: GaugeStation) -> str:
        if gauge.is_tidal:
            return "wl"
        return "00065"

    @staticmethod
    def _best_target_var(gauge: GaugeStation, aligned: pd.DataFrame) -> str:
        """Pick the best available variable for a gauge: stage preferred,
        discharge as fallback, water_level for tidal gauges."""
        if gauge.is_tidal:
            return "wl"
        # Prefer stage
        stage_col = f"{gauge.station_id}_00065"
        if stage_col in aligned.columns and aligned[stage_col].notna().sum() > 30:
            return "00065"
        # Fallback to discharge
        q_col = f"{gauge.station_id}_00060"
        if q_col in aligned.columns and aligned[q_col].notna().sum() > 30:
            return "00060"
        return "00065"  # default

    @staticmethod
    def _decay_weights(dates: pd.DatetimeIndex, half_life: int) -> np.ndarray:
        max_date = dates.max()
        days_ago = (max_date - dates).days
        return np.exp(-np.log(2) * days_ago / half_life)

    def _estimate_horizon_errors(
        self,
        aligned: pd.DataFrame,
        gauge: GaugeStation,
        tvar: str,
    ) -> None:
        """Estimate prediction error at each forecast horizon (1–28 days)
        using a rolling-origin scheme on the last 20 % of data."""
        sid = gauge.station_id
        tcol = f"{sid}_{tvar}"
        if tcol not in aligned.columns or sid not in self.models:
            return

        model = self.models[sid]
        n = len(aligned)
        val_start = int(n * 0.8)
        errors: Dict[int, List[float]] = {h: [] for h in range(1, FORECAST_HORIZON + 1)}

        # Simple 1-step error estimate; scale for multi-step
        try:
            X, y = self.feat_eng.build_training_set(aligned, sid, tvar)
            if len(X) < val_start + 10:
                return
            X_val = X.iloc[val_start:]
            y_val = y.iloc[val_start:]
            preds = model.predict(X_val)
            residuals = np.abs(y_val.values - preds)
            base_err = float(np.median(residuals))
        except Exception:
            base_err = 1.0

        # Store horizon-scaled errors
        self.horizon_errors[sid] = {
            h: base_err * np.sqrt(h) for h in range(1, FORECAST_HORIZON + 1)
        }


# ═══════════════════════════════════════════════════════════════════════════
# Anomaly detection
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class AnomalyFlag:
    """Container for one anomaly detection result."""

    station_id: str
    station_name: str
    flag_type: str  # "datum_drift", "outlier", "flatline", "spike"
    severity: str  # "warning", "critical"
    message: str
    date_detected: str
    value: float = np.nan
    expected: float = np.nan


class AnomalyDetector:
    """Detect faulty gauges by comparing each gauge against its
    network-predicted value and checking for suspicious patterns."""

    def __init__(self, network: List[GaugeStation] | None = None):
        self.network = network or GAUGE_NETWORK
        self._gauge_map = {g.station_id: g for g in self.network}

    # ------------------------------------------------------------------
    def detect(
        self,
        aligned: pd.DataFrame,
        forecaster: NetworkForecaster | None = None,
    ) -> List[AnomalyFlag]:
        """Run all anomaly checks on the recent aligned data."""
        flags: List[AnomalyFlag] = []
        flags.extend(self._check_flatlines(aligned))
        flags.extend(self._check_spikes(aligned))
        if forecaster is not None and forecaster.models:
            flags.extend(self._check_network_consistency(aligned, forecaster))
            flags.extend(self._check_datum_drift(aligned, forecaster))
        return flags

    # ------------------------------------------------------------------
    def _check_flatlines(self, aligned: pd.DataFrame) -> List[AnomalyFlag]:
        """Flag gauges reporting the exact same value for ≥5 consecutive days."""
        flags = []
        for sid, gauge in self._gauge_map.items():
            for suffix in ["00065", "wl"]:
                col = f"{sid}_{suffix}"
                if col not in aligned.columns:
                    continue
                s = aligned[col].dropna()
                if len(s) < 5:
                    continue
                # Count consecutive identical values at the tail
                tail = s.iloc[-10:]
                last_val = tail.iloc[-1]
                n_same = 0
                for v in reversed(tail.values):
                    if v == last_val:
                        n_same += 1
                    else:
                        break
                if n_same >= 5:
                    flags.append(AnomalyFlag(
                        station_id=sid,
                        station_name=gauge.name,
                        flag_type="flatline",
                        severity="warning",
                        message=(
                            f"Stage has been exactly {last_val:.2f} ft for "
                            f"{n_same} consecutive days – possible sensor freeze."
                        ),
                        date_detected=str(s.index[-1].date()),
                        value=last_val,
                    ))
        return flags

    # ------------------------------------------------------------------
    def _check_spikes(self, aligned: pd.DataFrame) -> List[AnomalyFlag]:
        """Flag unrealistic single-day jumps (> 5 × rolling σ)."""
        flags = []
        for sid, gauge in self._gauge_map.items():
            for suffix in ["00065", "wl"]:
                col = f"{sid}_{suffix}"
                if col not in aligned.columns:
                    continue
                s = aligned[col].dropna()
                if len(s) < 30:
                    continue
                diff = s.diff().abs()
                roll_std = diff.rolling(28, min_periods=7).std()
                last_diff = diff.iloc[-1]
                last_std = roll_std.iloc[-1]
                if last_std > 0 and last_diff > 5 * last_std:
                    flags.append(AnomalyFlag(
                        station_id=sid,
                        station_name=gauge.name,
                        flag_type="spike",
                        severity="critical",
                        message=(
                            f"Single-day change of {last_diff:.2f} ft exceeds "
                            f"5× the 28-day rolling σ ({last_std:.2f} ft)."
                        ),
                        date_detected=str(s.index[-1].date()),
                        value=float(s.iloc[-1]),
                    ))
        return flags

    # ------------------------------------------------------------------
    def _check_network_consistency(
        self,
        aligned: pd.DataFrame,
        forecaster: NetworkForecaster,
    ) -> List[AnomalyFlag]:
        """Compare recent observations against the network-predicted value.
        A large residual indicates the gauge may be malfunctioning or its
        datum has shifted."""
        flags = []
        recent = aligned.iloc[-RECENT_DAYS:]

        for sid, model in forecaster.models.items():
            gauge = self._gauge_map.get(sid)
            if gauge is None:
                continue
            tvar = forecaster.model_target_vars.get(sid, NetworkForecaster._target_var(gauge))
            tcol = f"{sid}_{tvar}"
            if tcol not in recent.columns:
                continue

            try:
                X, y = forecaster.feat_eng.build_training_set(recent, sid, tvar)
            except Exception:
                continue
            if len(X) < 5:
                continue

            preds = model.predict(X)
            residuals = y.values - preds
            if len(residuals) < 5:
                continue
            mu = np.mean(residuals)
            sigma = np.std(residuals) if np.std(residuals) > 0 else 1.0
            last_residual = residuals[-1]
            zscore = (last_residual - mu) / sigma

            if abs(zscore) > ANOMALY_ZSCORE_THRESHOLD:
                sev = "critical" if abs(zscore) > 5 else "warning"
                flags.append(AnomalyFlag(
                    station_id=sid,
                    station_name=gauge.name if gauge else sid,
                    flag_type="outlier",
                    severity=sev,
                    message=(
                        f"Latest observation deviates {zscore:+.1f}σ from the "
                        f"network-predicted value (observed {y.values[-1]:.2f}, "
                        f"predicted {preds[-1]:.2f})."
                    ),
                    date_detected=str(y.index[-1].date()),
                    value=float(y.values[-1]),
                    expected=float(preds[-1]),
                ))
        return flags

    # ------------------------------------------------------------------
    def _check_datum_drift(
        self,
        aligned: pd.DataFrame,
        forecaster: NetworkForecaster,
    ) -> List[AnomalyFlag]:
        """Detect gradual datum drift by looking at the trend in residuals
        over a sliding window."""
        flags = []
        recent = aligned.iloc[-RECENT_DAYS:]

        for sid, model in forecaster.models.items():
            gauge = self._gauge_map.get(sid)
            if gauge is None:
                continue
            tvar = forecaster.model_target_vars.get(sid, NetworkForecaster._target_var(gauge))
            tcol = f"{sid}_{tvar}"
            if tcol not in recent.columns:
                continue

            try:
                X, y = forecaster.feat_eng.build_training_set(recent, sid, tvar)
            except Exception:
                continue
            if len(X) < DRIFT_WINDOW_DAYS * 2:
                continue

            preds = model.predict(X)
            residuals = pd.Series(y.values - preds, index=y.index)

            # Compare mean residual in last DRIFT_WINDOW vs. earlier
            recent_mean = residuals.iloc[-DRIFT_WINDOW_DAYS:].mean()
            earlier_mean = residuals.iloc[:-DRIFT_WINDOW_DAYS].mean()
            earlier_std = residuals.iloc[:-DRIFT_WINDOW_DAYS].std()
            if earlier_std == 0:
                continue

            shift = (recent_mean - earlier_mean) / earlier_std

            if abs(shift) > DRIFT_SHIFT_THRESHOLD:
                unit = "cfs" if tvar == "00060" else "ft"
                flags.append(AnomalyFlag(
                    station_id=sid,
                    station_name=gauge.name if gauge else sid,
                    flag_type="datum_drift",
                    severity="warning",
                    message=(
                        f"Mean prediction residual has shifted by {shift:+.1f}s "
                        f"over the last {DRIFT_WINDOW_DAYS} days "
                        f"(from {earlier_mean:+.2f} to {recent_mean:+.2f} {unit}). "
                        f"Possible datum drift or systematic equipment offset."
                    ),
                    date_detected=str(residuals.index[-1].date()),
                    value=float(recent_mean),
                    expected=0.0,
                ))
        return flags


# ═══════════════════════════════════════════════════════════════════════════
# River-mile interpolation & coordinate conversion
# ═══════════════════════════════════════════════════════════════════════════
class RiverMileMapper:
    """Convert between coordinates and LMR river miles, and interpolate
    forecasts at arbitrary river miles between gauge stations."""

    def __init__(self, csv_path: str | Path = "river_miles.csv"):
        self.csv_path = Path(csv_path)
        self._df: pd.DataFrame | None = None

    @property
    def data(self) -> pd.DataFrame:
        if self._df is None:
            self._load()
        return self._df  # type: ignore[return-value]

    def _load(self) -> None:
        if not self.csv_path.exists():
            log.warning("river_miles.csv not found at %s", self.csv_path)
            self._df = pd.DataFrame()
            return
        df = pd.read_csv(self.csv_path)
        # Keep only Lower Mississippi rows
        mask = (df["RIVER_CODE"] == "MI") & (df["RIVER_NAME"].str.contains("LO", case=False, na=False))
        self._df = df.loc[mask].copy()
        self._df["MILE"] = pd.to_numeric(self._df["MILE"], errors="coerce")
        self._df = self._df.dropna(subset=["MILE", "LATITUDE1", "LONGITUDE1"])
        self._df = self._df.sort_values("MILE").reset_index(drop=True)
        log.debug("Loaded %d Lower-Mississippi river-mile points.", len(self._df))

    # ------------------------------------------------------------------
    def coords_to_mile(self, lat: float, lon: float) -> float:
        """Return the nearest LMR river mile to the given coordinates."""
        d = self.data
        if d.empty:
            raise ValueError("No river-mile data loaded")
        # Haversine-like fast approximation (valid for short distances)
        dlat = d["LATITUDE1"].values - lat
        dlon = (d["LONGITUDE1"].values - lon) * np.cos(np.radians(lat))
        dist = dlat**2 + dlon**2
        idx = np.argmin(dist)
        return float(d["MILE"].iloc[idx])

    def mile_to_coords(self, mile: float) -> Tuple[float, float]:
        """Interpolate lat/lon for a given river mile."""
        d = self.data
        if d.empty:
            raise ValueError("No river-mile data loaded")
        # Find bracketing points
        below = d[d["MILE"] <= mile]
        above = d[d["MILE"] >= mile]
        if below.empty:
            row = above.iloc[0]
            return float(row["LATITUDE1"]), float(row["LONGITUDE1"])
        if above.empty:
            row = below.iloc[-1]
            return float(row["LATITUDE1"]), float(row["LONGITUDE1"])
        lo = below.iloc[-1]
        hi = above.iloc[0]
        if lo["MILE"] == hi["MILE"]:
            return float(lo["LATITUDE1"]), float(lo["LONGITUDE1"])
        frac = (mile - lo["MILE"]) / (hi["MILE"] - lo["MILE"])
        lat = lo["LATITUDE1"] + frac * (hi["LATITUDE1"] - lo["LATITUDE1"])
        lon = lo["LONGITUDE1"] + frac * (hi["LONGITUDE1"] - lo["LONGITUDE1"])
        return float(lat), float(lon)

    # ------------------------------------------------------------------
    def interpolate_forecast(
        self,
        forecasts: pd.DataFrame,
        target_mile: float,
        network: List[GaugeStation] | None = None,
        model_target_vars: Dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Interpolate forecasted stage at *target_mile* from gauge-level
        forecasts using distance-weighted interpolation between the two
        nearest bracketing mainstem gauges.

        Only uses gauges predicting stage (00065) or water level for
        interpolation.  If a gauge at the exact target mile predicts
        discharge (00060), both discharge and stage (via interpolation from
        neighbors) are returned.

        Returns a DataFrame with columns ``stage_pred``, ``stage_lo``,
        ``stage_hi``, and optionally ``discharge_pred``, ``discharge_lo``,
        ``discharge_hi``.
        """
        if network is None:
            network = GAUGE_NETWORK
        if model_target_vars is None:
            model_target_vars = {}

        # Separate mainstem gauges by variable type
        stage_gauges = []
        discharge_gauges = []
        for g in network:
            col = f"{g.station_id}_pred"
            if col not in forecasts.columns:
                continue
            if "mississippi" not in g.river:
                continue
            tvar = model_target_vars.get(g.station_id, "00065")
            if tvar in ("00065", "wl"):
                stage_gauges.append(g)
            elif tvar == "00060":
                discharge_gauges.append(g)
        stage_gauges.sort(key=lambda g: g.river_mile)
        discharge_gauges.sort(key=lambda g: g.river_mile)

        out = pd.DataFrame(index=forecasts.index)
        out.index.name = "date"

        # --- Stage interpolation (preferred) ---
        if stage_gauges:
            stage_out = self._interpolate_variable(
                forecasts, target_mile, stage_gauges, "stage"
            )
            for c in stage_out.columns:
                out[c] = stage_out[c]
        else:
            log.warning("No mainstem stage gauges available for interpolation.")

        # --- Check if a discharge gauge sits at (or near) the target mile ---
        for g in discharge_gauges:
            if abs(g.river_mile - target_mile) < 20:
                for suffix in ["pred", "lo", "hi"]:
                    col = f"{g.station_id}_{suffix}"
                    if col in forecasts.columns:
                        out[f"discharge_{suffix}"] = forecasts[col]
                break  # use nearest one only

        return out

    # ------------------------------------------------------------------
    def _interpolate_variable(
        self,
        forecasts: pd.DataFrame,
        target_mile: float,
        gauges: List[GaugeStation],
        prefix: str,
    ) -> pd.DataFrame:
        """Linear interpolation between two bracketing gauges."""
        below = [g for g in gauges if g.river_mile <= target_mile]
        above = [g for g in gauges if g.river_mile >= target_mile]

        if not below and not above:
            return pd.DataFrame()
        if not below:
            return self._extract_gauge_forecast(forecasts, above[0], prefix)
        if not above:
            return self._extract_gauge_forecast(forecasts, below[-1], prefix)

        g_lo = below[-1]
        g_hi = above[0]

        if g_lo.station_id == g_hi.station_id:
            return self._extract_gauge_forecast(forecasts, g_lo, prefix)

        span = g_hi.river_mile - g_lo.river_mile
        frac = (target_mile - g_lo.river_mile) / span if span > 0 else 0.5

        out = pd.DataFrame(index=forecasts.index)
        for suffix in ["pred", "lo", "hi"]:
            col_lo = f"{g_lo.station_id}_{suffix}"
            col_hi = f"{g_hi.station_id}_{suffix}"
            s_lo = forecasts.get(col_lo, pd.Series(np.nan, index=forecasts.index))
            s_hi = forecasts.get(col_hi, pd.Series(np.nan, index=forecasts.index))
            out[f"{prefix}_{suffix}"] = s_lo * (1 - frac) + s_hi * frac
        return out

    @staticmethod
    def _extract_gauge_forecast(
        forecasts: pd.DataFrame, gauge: GaugeStation, prefix: str = "stage"
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=forecasts.index)
        sid = gauge.station_id
        out[f"{prefix}_pred"] = forecasts.get(f"{sid}_pred", np.nan)
        out[f"{prefix}_lo"] = forecasts.get(f"{sid}_lo", np.nan)
        out[f"{prefix}_hi"] = forecasts.get(f"{sid}_hi", np.nan)
        out.index.name = "date"
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Main forecaster class
# ═══════════════════════════════════════════════════════════════════════════
class LMRForecaster:
    """Top-level orchestrator: fetch data → train models → forecast →
    detect anomalies → interpolate to any river mile."""

    def __init__(
        self,
        data_dir: str | Path = ".",
        cache_dir: str | Path = CACHE_DIR,
        model_dir: str | Path = MODEL_DIR,
        network: List[GaugeStation] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.network = network or GAUGE_NETWORK
        self.data_mgr = DataManager(self.network, Path(cache_dir))
        self.forecaster = NetworkForecaster(self.network, Path(model_dir))
        self.anomaly_det = AnomalyDetector(self.network)
        self.mapper = RiverMileMapper(self.data_dir / "river_miles.csv")
        self._aligned: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def fetch_recent(self, days: int = RECENT_DAYS) -> Dict[str, pd.DataFrame]:
        """Fetch recent observations for all gauges."""
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        return self.data_mgr.fetch_all(
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )

    def fetch_historical(self, years: int = HISTORICAL_YEARS_DEFAULT) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for model training."""
        end = datetime.utcnow()
        start = end - timedelta(days=years * 365)
        return self.data_mgr.fetch_all(
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )

    # ------------------------------------------------------------------
    def train(self, years: int = HISTORICAL_YEARS_DEFAULT) -> None:
        """Fetch historical data and train all models."""
        log.info("═" * 60)
        log.info("TRAINING – fetching %d years of historical data …", years)
        log.info("═" * 60)
        raw = self.fetch_historical(years)
        aligned = self.data_mgr.build_aligned_daily(raw)
        if aligned.empty:
            log.error("No data available – cannot train.")
            return
        log.info(
            "Aligned dataset: %d rows × %d columns (%s → %s)",
            len(aligned), aligned.shape[1],
            aligned.index[0].date(), aligned.index[-1].date(),
        )
        self._aligned = aligned
        self.forecaster.train_all(aligned)

    # ------------------------------------------------------------------
    def forecast(
        self,
        river_mile: float | None = None,
        lat: float | None = None,
        lon: float | None = None,
        days: int = FORECAST_HORIZON,
    ) -> pd.DataFrame:
        """Generate a forecast.

        Specify location by *river_mile* **or** *lat*/*lon* (which will be
        converted to the nearest river mile).

        Returns a DataFrame with ``stage_pred``, ``stage_lo``, ``stage_hi``
        indexed by date.
        """
        # Resolve location
        if river_mile is None and lat is not None and lon is not None:
            river_mile = self.mapper.coords_to_mile(lat, lon)
            log.info("Coordinates (%.4f, %.4f) → LMR river mile %.1f", lat, lon, river_mile)
        if river_mile is None:
            river_mile = 437.0  # default: Vicksburg
            log.info("No location specified – defaulting to Vicksburg (RM 437)")

        # Ensure models are loaded
        if not self.forecaster.models:
            loaded = self.forecaster.load_models()
            if loaded == 0:
                log.error(
                    "No trained models available.  Run with --train first."
                )
                return pd.DataFrame()

        # Fetch recent data
        log.info("Fetching recent %d-day observations …", RECENT_DAYS)
        raw = self.fetch_recent(RECENT_DAYS)
        aligned = self.data_mgr.build_aligned_daily(raw)
        if aligned.empty:
            log.error("No recent data available – cannot forecast.")
            return pd.DataFrame()
        self._aligned = aligned

        log.info("Generating %d-day recursive forecast …", days)
        gauge_forecasts = self.forecaster.forecast(aligned, days)

        # Interpolate at target river mile
        result = self.mapper.interpolate_forecast(
            gauge_forecasts, river_mile, self.network,
            model_target_vars=self.forecaster.model_target_vars,
        )

        if result.empty:
            log.warning(
                "Could not interpolate to RM %.1f – returning raw gauge forecasts.",
                river_mile,
            )
            return gauge_forecasts

        return result

    # ------------------------------------------------------------------
    def check_anomalies(self) -> List[AnomalyFlag]:
        """Fetch recent data and run anomaly detection."""
        if not self.forecaster.models:
            self.forecaster.load_models()

        raw = self.fetch_recent(RECENT_DAYS)
        aligned = self.data_mgr.build_aligned_daily(raw)
        if aligned.empty:
            log.error("No recent data for anomaly check.")
            return []
        self._aligned = aligned
        flags = self.anomaly_det.detect(aligned, self.forecaster)
        return flags

    # ------------------------------------------------------------------
    def get_gauge_forecasts(self, days: int = FORECAST_HORIZON) -> pd.DataFrame:
        """Return raw per-gauge forecasts (useful for inspection)."""
        if self._aligned is None:
            raw = self.fetch_recent(RECENT_DAYS)
            self._aligned = self.data_mgr.build_aligned_daily(raw)
        if not self.forecaster.models:
            self.forecaster.load_models()
        return self.forecaster.forecast(self._aligned, days)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Return a human-readable summary of the gauge network status."""
        lines = ["Lower Mississippi River Forecaster - Gauge Network Summary"]
        lines.append("=" * 62)
        for tier_name, tier_id in [
            ("Gulf / Tidal", 0),
            ("Tributaries", 1),
            ("Upper Mainstem", 2),
            ("Lower Mainstem", 3),
        ]:
            tier_gauges = [g for g in self.network if g.forecast_tier == tier_id]
            if not tier_gauges:
                continue
            lines.append(f"\n-- {tier_name} --")
            for g in sorted(tier_gauges, key=lambda x: -x.river_mile):
                status = "+" if g.station_id in self.forecaster.models else "-"
                mae_str = ""
                if g.station_id in self.forecaster.models:
                    mae = self.forecaster.models[g.station_id].val_mae
                    tvar = self.forecaster.model_target_vars.get(g.station_id, "")
                    unit = "cfs" if tvar == "00060" else "ft"
                    mae_str = f"  (val MAE {mae:.2f} {unit}, var={tvar})"
                lines.append(
                    f"  [{status}] {g.name:<45s} RM {g.river_mile:>6.0f}{mae_str}"
                )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation helpers (optional – requires matplotlib)
# ═══════════════════════════════════════════════════════════════════════════
def plot_forecast(
    forecast_df: pd.DataFrame,
    title: str = "LMR Stage Forecast",
    save_path: str | None = None,
) -> None:
    """Plot the interpolated forecast with confidence bands."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        log.warning("matplotlib not installed – skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    dates = forecast_df.index

    if "stage_pred" in forecast_df.columns:
        ax.plot(dates, forecast_df["stage_pred"], "b-o", markersize=3, label="Predicted stage")
        if "stage_lo" in forecast_df.columns and "stage_hi" in forecast_df.columns:
            ax.fill_between(
                dates,
                forecast_df["stage_lo"],
                forecast_df["stage_hi"],
                alpha=0.2,
                color="blue",
                label="Confidence band",
            )
    else:
        # Plot all _pred columns
        for col in forecast_df.columns:
            if col.endswith("_pred"):
                ax.plot(dates, forecast_df[col], "-o", markersize=2, label=col)

    ax.set_xlabel("Date")
    ax.set_ylabel("Stage (ft)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        log.info("Plot saved to %s", save_path)
    else:
        plt.show()


def plot_gauge_network(
    gauge_forecasts: pd.DataFrame,
    network: List[GaugeStation] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot forecasts for all mainstem gauges on a single figure."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        log.warning("matplotlib not installed – skipping plot.")
        return

    if network is None:
        network = GAUGE_NETWORK

    mainstem = sorted(
        [g for g in network if "mississippi" in g.river],
        key=lambda g: -g.river_mile,
    )

    n = len(mainstem)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, g in zip(axes, mainstem):
        pcol = f"{g.station_id}_pred"
        if pcol in gauge_forecasts.columns:
            ax.plot(gauge_forecasts.index, gauge_forecasts[pcol], "b-", linewidth=1.2)
            lo_col = f"{g.station_id}_lo"
            hi_col = f"{g.station_id}_hi"
            if lo_col in gauge_forecasts.columns:
                ax.fill_between(
                    gauge_forecasts.index,
                    gauge_forecasts[lo_col],
                    gauge_forecasts[hi_col],
                    alpha=0.15, color="blue",
                )
        ax.set_ylabel("ft")
        ax.set_title(f"{g.name} (RM {g.river_mile:.0f})", fontsize=9, loc="left")
        ax.grid(alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.suptitle("Lower Mississippi River – Gauge Network Forecast", fontsize=12)
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        log.info("Network plot saved to %s", save_path)
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════
def print_forecast_table(df: pd.DataFrame) -> None:
    """Pretty-print a forecast DataFrame to the console."""
    has_stage = "stage_pred" in df.columns
    has_discharge = "discharge_pred" in df.columns

    if has_stage and has_discharge:
        print(f"\n{'Date':<12s} {'Stage(ft)':>10s} {'Low':>10s} {'High':>10s}"
              f" {'Q(cfs)':>12s} {'Low':>12s} {'High':>12s}")
        print("-" * 80)
        for date, row in df.iterrows():
            d = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            print(
                f"{d:<12s} {row.get('stage_pred', np.nan):>10.2f} "
                f"{row.get('stage_lo', np.nan):>10.2f} {row.get('stage_hi', np.nan):>10.2f}"
                f" {row.get('discharge_pred', np.nan):>12.0f} "
                f"{row.get('discharge_lo', np.nan):>12.0f} {row.get('discharge_hi', np.nan):>12.0f}"
            )
    elif has_stage:
        print(f"\n{'Date':<12s} {'Stage(ft)':>10s} {'Low':>10s} {'High':>10s}")
        print("-" * 44)
        for date, row in df.iterrows():
            d = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            print(
                f"{d:<12s} {row.get('stage_pred', np.nan):>10.2f} "
                f"{row.get('stage_lo', np.nan):>10.2f} {row.get('stage_hi', np.nan):>10.2f}"
            )
    elif has_discharge:
        print(f"\n{'Date':<12s} {'Q(cfs)':>12s} {'Low':>12s} {'High':>12s}")
        print("-" * 50)
        for date, row in df.iterrows():
            d = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            print(
                f"{d:<12s} {row.get('discharge_pred', np.nan):>12.0f} "
                f"{row.get('discharge_lo', np.nan):>12.0f} {row.get('discharge_hi', np.nan):>12.0f}"
            )
    else:
        print(df.to_string())


def print_anomaly_report(flags: List[AnomalyFlag]) -> None:
    """Pretty-print anomaly flags."""
    if not flags:
        print("\n[OK] No anomalies detected in the gauge network.")
        return
    print(f"\n[!] {len(flags)} anomaly flag(s) detected:")
    print("-" * 70)
    for f in flags:
        sev_icon = "[CRIT]" if f.severity == "critical" else "[WARN]"
        print(f"{sev_icon} [{f.flag_type.upper()}] {f.station_name} ({f.station_id})")
        print(f"   Date: {f.date_detected}")
        print(f"   {f.message}")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="LMR_Forecaster",
        description=(
            "Lower Mississippi River stage forecaster.  Predicts stage at "
            "any river mile for up to 28 days using USGS/NOAA gauge data "
            "and XGBoost machine learning."
        ),
    )
    loc = p.add_argument_group("Location (specify one)")
    loc.add_argument(
        "--mile", type=float, default=None,
        help="Target LMR river mile (Head of Passes = 0, Cairo ≈ 953). "
             "Default: 437 (Vicksburg).",
    )
    loc.add_argument("--lat", type=float, default=None, help="Latitude (decimal degrees)")
    loc.add_argument("--lon", type=float, default=None, help="Longitude (decimal degrees)")

    p.add_argument(
        "--days", type=int, default=FORECAST_HORIZON,
        help=f"Forecast horizon in days (default {FORECAST_HORIZON}).",
    )

    p.add_argument(
        "--train", action="store_true",
        help="(Re)train models on historical data before forecasting.",
    )
    p.add_argument(
        "--years", type=int, default=HISTORICAL_YEARS_DEFAULT,
        help=f"Years of historical data for training (default {HISTORICAL_YEARS_DEFAULT}).",
    )

    p.add_argument(
        "--check-anomalies", action="store_true",
        help="Run anomaly detection on the gauge network.",
    )

    p.add_argument(
        "--plot", action="store_true",
        help="Show forecast plot (requires matplotlib).",
    )
    p.add_argument(
        "--plot-network", action="store_true",
        help="Show per-gauge network forecast plot.",
    )
    p.add_argument(
        "--save-plot", type=str, default=None,
        help="Save forecast plot to file instead of showing.",
    )

    p.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save forecast to CSV file.",
    )

    p.add_argument(
        "--summary", action="store_true",
        help="Print gauge-network summary and exit.",
    )

    p.add_argument(
        "--clear-cache", action="store_true",
        help="Clear cached data and exit.",
    )

    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("LMR_Forecaster").setLevel(logging.DEBUG)

    forecaster = LMRForecaster()

    # ------------------------------------------------------------------
    if args.clear_cache:
        forecaster.data_mgr.cache.clear()
        print("Cache cleared.")
        return

    if args.summary:
        forecaster.forecaster.load_models()
        print(forecaster.summary())
        return

    # ------------------------------------------------------------------
    if args.train:
        forecaster.train(years=args.years)
        print(forecaster.summary())

    # ------------------------------------------------------------------
    if args.check_anomalies:
        flags = forecaster.check_anomalies()
        print_anomaly_report(flags)
        if not args.train and args.mile is None and args.lat is None:
            return  # anomaly check only

    # ------------------------------------------------------------------
    # Forecast
    target_mile = args.mile
    if target_mile is None and args.lat is not None and args.lon is not None:
        target_mile = forecaster.mapper.coords_to_mile(args.lat, args.lon)
        log.info("Coordinates → RM %.1f", target_mile)

    if target_mile is None and not args.train and not args.check_anomalies:
        target_mile = 437.0  # default Vicksburg

    if target_mile is not None:
        result = forecaster.forecast(river_mile=target_mile, days=args.days)
        if result.empty:
            print("No forecast produced.  Run with --train first if models "
                  "are not yet built.")
            return

        lat, lon = forecaster.mapper.mile_to_coords(target_mile)
        print(f"\n{'=' * 60}")
        print(f"  Lower Mississippi River Stage Forecast")
        print(f"  River Mile {target_mile:.1f}  ({lat:.4f} N, {lon:.4f} W)")
        print(f"  Horizon: {args.days} days")
        print(f"{'=' * 60}")
        print_forecast_table(result)

        # Anomaly check (always run alongside forecast)
        flags = forecaster.check_anomalies()
        if flags:
            print_anomaly_report(flags)

        if args.output:
            result.to_csv(args.output)
            print(f"\nForecast saved to {args.output}")

        if args.plot or args.save_plot:
            plot_forecast(
                result,
                title=f"LMR Forecast – RM {target_mile:.0f}",
                save_path=args.save_plot,
            )

        if args.plot_network:
            gf = forecaster.get_gauge_forecasts(args.days)
            plot_gauge_network(gf, save_path=args.save_plot)


if __name__ == "__main__":
    main()
