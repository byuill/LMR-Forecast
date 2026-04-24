#!/usr/bin/env python3
"""
LMR_doctor.py
=============

Machine-learning QAQC system for Lower Mississippi River stage data.

What it does:
1) Pulls historical stage data from RiverGages stations listed in RG_Stations.csv.
2) Pulls discharge data from USGS stations listed in USGS_Stations.csv.
3) Pulls Gulf base-level signals from NOAA Grand Isle and Pensacola gauges.
4) Learns dynamic cross-station behavior using lagged network features.
5) Iteratively flags outliers, retrains after removing flagged points,
   and produces corrected + gap-filled stage series.
6) Saves artifacts and outputs for reproducible future use.
7) Provides a query dashboard to retrieve high-quality predicted stage data
   over any interval within the trained temporal domain.

Notes:
- Gage datum shift PDFs in GageNotes are inventoried and referenced.
  Automated PDF parsing is intentionally left out to avoid brittle OCR logic.
  If you maintain a datum event table later, it can be merged as features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from Rivergagescom_API import RiverGagesAPI
except Exception:
    RiverGagesAPI = None

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "qaqc"
MODEL_DIR = ROOT / "models" / "qaqc"
OUTPUT_DIR = ROOT / "models" / "qaqc_outputs"
GAGENOTES_DIR = ROOT / "GageNotes"
ARTIFACT_PATH = MODEL_DIR / "network_qaqc_artifact.pkl"

RG_STATIONS_FILE = ROOT / "RG_Stations.csv"
USGS_STATIONS_FILE = ROOT / "USGS_Stations.csv"
RIVER_MILES_FILE = ROOT / "river_miles.csv"

NOAA_STATIONS = {
    "grand_isle": "8761724",
    "pensacola": "8729840",
}

USGS_BASE = "https://waterservices.usgs.gov/nwis"
NOAA_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
VDATUM_BASE = "https://vdatum.noaa.gov/vdatumweb/api/convert"

FT_TO_M = 0.3048                  # feet → metres (exact)
M_TO_FT = 1.0 / FT_TO_M          # metres → feet
# assumed vertical datum for RiverGages stage data
DEFAULT_SOURCE_DATUM = "NAVD88"
VDATUM_OFFSET_CACHE = CACHE_DIR / "vdatum_offsets.json"

REQUEST_TIMEOUT = 90
CHUNK_DAYS = 90

DEFAULT_LOOKBACK_YEARS = 15
DEFAULT_FREQ = "hybrid"
MAX_LAG_DAYS = 7
ITER_MAX = 4
OUTLIER_Z = 3.5
# Gap-fill spatial validation: maximum allowed deviation (ft) between a model
# gap-fill prediction and the linear interpolation from adjacent network gages.
# Deviations above this are physically implausible and the prediction is replaced
# by the spatial estimate.  5 ft matches the LMR_surfer spike-detector threshold.
GAPFILL_SPATIAL_THRESHOLD_FT = 5.0

TARGET_STAGE_VARIABLE = "HG"
USGS_DISCHARGE_PARAM = "00060"

# Travel-time calibration constants
# ──────────────────────────────────────────────────────────────────────────────
# The flood wave on the LMR travels roughly 60–110 river miles per day depending
# on discharge.  The power-law exponent 0.4 is derived empirically for alluvial
# rivers (Leopold & Maddock, 1953): V ∝ Q^0.4.  The base speed corresponds to
# roughly median main-stem flow at Tarbert Landing (~700 000 cfs).
FLOW_SPEED_BASE_MI_PER_DAY = 72.0       # speed at Q_REF discharge
FLOW_SPEED_Q_REF_CFS = 700_000.0        # normalisation reference discharge
FLOW_SPEED_EXPONENT = 0.4               # V ∝ Q^exponent
MAX_XCORR_LAG_DAYS = 21                 # search window for cross-correlation lag
# days each side for local-extremum detection
PEAK_HALF_WINDOW_DAYS = 15
PEAK_MIN_PROM_SIGMA = 0.4               # min prominence relative to robust σ
PEAK_MATCH_MAX_DAYS = MAX_XCORR_LAG_DAYS  # max lag when matching peaks
DISCHARGE_BIN_LABELS = ["low", "medium_low", "medium_high", "high"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("LMR_QAQC")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class StationMeta:
    station_id: str
    source: str
    name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    river_mile: Optional[float] = None


@dataclass
class StationModelBundle:
    station_id: str
    feature_columns: List[str]
    model: HistGradientBoostingRegressor
    metrics: Dict[str, float]
    outlier_timestamps: List[str]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    for d in [CACHE_DIR, MODEL_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def dt_to_str(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y-%m-%d")


def hash_key(parts: Iterable[str]) -> str:
    raw = "|".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def date_chunks(start: pd.Timestamp, end: pd.Timestamp, chunk_days: int = CHUNK_DAYS) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def parse_iso(s: str) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(s))


def robust_sigma(values) -> float:
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    sigma = float(1.4826 * mad)
    # Enforce a physical minimum floor of 0.25 ft. Prevents the QAQC threshold
    # from collapsing to 0 on highly accurate models and prevents fallback to std dev.
    if np.isnan(sigma) or sigma < 0.25:
        return 0.25
    return sigma


def sanitize_station_id(station_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", station_id)


class ProgressTracker:
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description.ljust(18)
        self.current = 0
        self.start_time = time.time()

    def format_time(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def update(self, step: int = 1, item_name: str = "") -> None:
        self.current += step
        if self.total == 0:
            return
        elapsed = time.time() - self.start_time
        progress = self.current / self.total
        eta_sec = (elapsed / progress) - elapsed if progress > 0 else 0
        bar_len = 25
        filled = int(bar_len * progress)
        bar = "█" * filled + "-" * (bar_len - filled)
        percent = int(progress * 100)
        sys.stdout.write(
            f"\r{self.description} |{bar}| {percent:3d}% [{self.current}/{self.total}] ETA: {self.format_time(eta_sec)} | {item_name[:15].ljust(15)}")
        sys.stdout.flush()

    def finish(self) -> None:
        if self.total == 0:
            return
        sys.stdout.write(
            f"\r{self.description} |{'█' * 25}| 100% [{self.total}/{self.total}] Time: {self.format_time(time.time() - self.start_time)}".ljust(85) + "\n")
        sys.stdout.flush()


class RiverMileMapper:
    def __init__(self, csv_path: Path = RIVER_MILES_FILE):
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None

    @property
    def data(self) -> pd.DataFrame:
        if self._df is None:
            if not self.csv_path.exists():
                log.warning("river_miles.csv not found at %s", self.csv_path)
                self._df = pd.DataFrame()
            else:
                df = pd.read_csv(self.csv_path)
                mask = (df["RIVER_CODE"] == "MI") & (
                    df["RIVER_NAME"].str.contains("LO", case=False, na=False))
                self._df = df.loc[mask].copy()
                self._df["MILE"] = pd.to_numeric(
                    self._df["MILE"], errors="coerce")
                self._df = self._df.dropna(subset=["MILE", "LATITUDE1", "LONGITUDE1"]).sort_values(
                    "MILE").reset_index(drop=True)
        return self._df

    def coords_to_mile(self, lat: float, lon: float) -> float:
        d = self.data
        if d.empty:
            return np.nan
        dlat = d["LATITUDE1"].values - lat
        dlon = (d["LONGITUDE1"].values - lon) * np.cos(np.radians(lat))
        dist = dlat**2 + dlon**2
        idx = np.argmin(dist)
        return float(d["MILE"].iloc[idx])


# ---------------------------------------------------------------------------
# RiverGages data client
# ---------------------------------------------------------------------------
class RiverGagesClient:
    def __init__(self):
        self.api = RiverGagesAPI() if RiverGagesAPI else None
        self.base_url = "https://rivergages.mvr.usace.army.mil/watercontrol/webservices/rest/webserviceWaterML.cfc"

    def _request_raw(
            self,
            meth: str,
            site: str,
            variable: str,
            begin_date: str,
            end_date: str,
    ) -> str:
        if self.api is not None:
            data = self.api.query(
                meth=meth,
                site=site,
                variable=variable,
                begin_date=begin_date,
                end_date=end_date,
            )
            return data or ""

        params = {
            "method": "RGWML",
            "meth": meth,
            "site": site,
            "location": site,
            "variable": variable,
            "beginDate": begin_date,
            "endDate": end_date,
            "authtoken": "RiverGages",
            "authToken": "RiverGages",
        }
        r = requests.get(self.base_url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.text

    @staticmethod
    def _extract_timeseries(xml_text: str) -> pd.DataFrame:
        if not xml_text.strip():
            return pd.DataFrame(columns=["timestamp", "value"])

        timestamps: List[pd.Timestamp] = []
        values: List[float] = []

        pattern = re.compile(r"dateTime=\"([^\"]+)\"[^>]*>([^<]+)<")
        for match in pattern.finditer(xml_text):
            ts_raw, val_raw = match.groups()
            try:
                ts = pd.to_datetime(ts_raw, utc=False)
                val = float(val_raw)
                # Filter out USACE sensor error codes (e.g., -99, -999) and impossible elevations
                if val <= -50.0 or val >= 1000.0:
                    continue
            except Exception:
                continue
            timestamps.append(pd.Timestamp(ts).tz_localize(
                None) if getattr(ts, "tzinfo", None) else pd.Timestamp(ts))
            values.append(val)

        if not timestamps:
            return pd.DataFrame(columns=["timestamp", "value"])

        df = pd.DataFrame({"timestamp": timestamps, "value": values})
        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        return df

    def get_station_metadata(self, site: str) -> dict:
        try:
            raw = self._request_raw(
                "GetSiteInfo", site, TARGET_STAGE_VARIABLE, "2023-01-01T00:00", "2023-01-02T00:00")
            lat_m = re.search(
                r'<latitude>([^<]+)</latitude>', raw, re.IGNORECASE)
            lon_m = re.search(
                r'<longitude>([^<]+)</longitude>', raw, re.IGNORECASE)
            if lat_m and lon_m:
                return {"lat": float(lat_m.group(1)), "lon": float(lon_m.group(1))}
        except Exception as exc:
            log.warning("Metadata fetch failed for %s: %s", site, exc)
        return {}

    def fetch_stage(
            self,
            site: str,
            start: pd.Timestamp,
            end: pd.Timestamp,
            variable: str = TARGET_STAGE_VARIABLE,
    ) -> pd.Series:
        cache_name = f"rg_{site}_{dt_to_str(start)}_{dt_to_str(end)}_{variable}.csv"
        cache_path = CACHE_DIR / cache_name
        if cache_path.exists():
            cdf = pd.read_csv(cache_path, parse_dates=["timestamp"])
            return pd.Series(cdf["value"].values, index=cdf["timestamp"], name=site)

        chunk_frames: List[pd.DataFrame] = []
        for chunk_start, chunk_end in date_chunks(start, end):
            b = chunk_start.strftime("%Y-%m-%dT00:00")
            e = chunk_end.strftime("%Y-%m-%dT23:59")
            try:
                raw = self._request_raw("GetValues", site, variable, b, e)
                df = self._extract_timeseries(raw)
                if not df.empty:
                    chunk_frames.append(df)
            except Exception as exc:
                log.warning(
                    "RiverGages fetch failed for site=%s chunk=%s..%s: %s", site, b, e, exc)

        if not chunk_frames:
            return pd.Series(dtype=float, name=site)

        merged = pd.concat(chunk_frames, ignore_index=True)
        merged = merged.sort_values("timestamp").drop_duplicates("timestamp")
        merged.to_csv(cache_path, index=False)
        return pd.Series(merged["value"].values, index=merged["timestamp"], name=site)


# ---------------------------------------------------------------------------
# USGS and NOAA clients
# ---------------------------------------------------------------------------
class USGSClient:
    @staticmethod
    def fetch_discharge(station_id: str, start: pd.Timestamp, end: pd.Timestamp, freq: str = "D") -> pd.Series:
        endpoint = "iv" if freq in ("H", "h") else "dv"
        cache_name = f"usgs_q_{endpoint}_{station_id}_{dt_to_str(start)}_{dt_to_str(end)}.csv"
        cache_path = CACHE_DIR / cache_name
        if cache_path.exists():
            cdf = pd.read_csv(cache_path, parse_dates=["date"])
            return pd.Series(cdf["value"].values, index=cdf["date"], name=f"q_{station_id}")

        rows: List[Tuple[pd.Timestamp, float]] = []
        chunk_days = 30 if freq in ("H", "h") else 3650
        for chunk_start, chunk_end in date_chunks(start, end, chunk_days=chunk_days):
            params = {
                "format": "json",
                "sites": station_id,
                "startDT": dt_to_str(chunk_start),
                "endDT": dt_to_str(chunk_end),
                "parameterCd": USGS_DISCHARGE_PARAM,
                "siteStatus": "all",
            }
            url = f"{USGS_BASE}/{endpoint}/"
            try:
                r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                payload = r.json()
                
                series = payload.get("value", {}).get("timeSeries", [])
                for ts in series:
                    values = ts.get("values", [])
                    for group in values:
                        for item in group.get("value", []):
                            d = item.get("dateTime")
                            v = item.get("value")
                            if d is None or v in (None, ""):
                                continue
                            try:
                                dt_val = pd.to_datetime(d).tz_localize(None)
                                fv = float(v)
                                if fv <= -9999.0:
                                    continue
                                if freq in ("D", "d"):
                                    dt_val = dt_val.normalize()
                                else:
                                    dt_val = dt_val.floor("h")
                                rows.append((dt_val, fv))
                            except Exception:
                                continue
            except Exception as exc:
                log.warning("USGS fetch failed station=%s chunk=%s..%s: %s", station_id, chunk_start, chunk_end, exc)

        if not rows:
            return pd.Series(dtype=float, name=f"q_{station_id}")

        rdf = pd.DataFrame(rows, columns=["date", "value"])
        rdf = rdf.groupby("date", as_index=False).mean().sort_values("date")
        rdf.to_csv(cache_path, index=False)
        return pd.Series(rdf["value"].values, index=rdf["date"], name=f"q_{station_id}")


class NOAAClient:
    @staticmethod
    def fetch_water_level(station_id: str, start: pd.Timestamp, end: pd.Timestamp, freq: str = "D") -> pd.Series:
        cache_name = f"noaa_wl_{freq}_{station_id}_{dt_to_str(start)}_{dt_to_str(end)}.csv"
        cache_path = CACHE_DIR / cache_name
        if cache_path.exists():
            cdf = pd.read_csv(cache_path, parse_dates=["date"])
            return pd.Series(cdf["value"].values, index=cdf["date"], name=f"wl_{station_id}")

        rows: List[Tuple[pd.Timestamp, float]] = []
        for chunk_start, chunk_end in date_chunks(start, end, chunk_days=31):
            params = {
                "product": "water_level",
                "application": "LMR_QAQC",
                "begin_date": chunk_start.strftime("%Y%m%d"),
                "end_date": chunk_end.strftime("%Y%m%d"),
                "datum": "MSL",
                "station": station_id,
                "time_zone": "gmt",
                "units": "metric",
                "format": "csv",
                "interval": "h",
            }
            try:
                r = requests.get(NOAA_BASE, params=params,
                                 timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                if "Date Time" not in r.text:
                    continue
                temp = pd.read_csv(pd.io.common.StringIO(r.text))
                if "Date Time" not in temp.columns or " Water Level" not in temp.columns:
                    continue
                temp = temp[["Date Time", " Water Level"]].copy()
                temp.columns = ["dt", "wl"]
                temp["dt"] = pd.to_datetime(temp["dt"], errors="coerce")
                temp["wl"] = pd.to_numeric(temp["wl"], errors="coerce")
                temp = temp.dropna(subset=["dt", "wl"])
                # Filter out NOAA error codes and physical impossibilities (units are in metric here)
                temp = temp[(temp["wl"] > -50.0) & (temp["wl"] < 100.0)]
                if freq in ("D", "d"):
                    temp["date"] = temp["dt"].dt.normalize()
                else:
                    temp["date"] = temp["dt"].dt.floor("h")
                grouped = temp.groupby("date", as_index=False)["wl"].mean()
                rows.extend(list(grouped.itertuples(index=False, name=None)))
            except Exception as exc:
                log.warning("NOAA fetch failed station=%s chunk=%s..%s: %s",
                            station_id, chunk_start, chunk_end, exc)

        if not rows:
            return pd.Series(dtype=float, name=f"wl_{station_id}")

        rdf = pd.DataFrame(rows, columns=["date", "value"]).drop_duplicates(
            "date").sort_values("date")
        rdf.to_csv(cache_path, index=False)
        return pd.Series(rdf["value"].values, index=rdf["date"], name=f"wl_{station_id}")


# ---------------------------------------------------------------------------
# GageNotes PDF parser — datum calibration adjustment history
# ---------------------------------------------------------------------------
GAGENOTES_ADJ_CACHE = CACHE_DIR / "gagenotes_adjustments.json"
# Per-station plain-text archive files live here (one file per PDF/station).
# These are the primary persistent store; the JSON above is a compiled summary.
DATUM_RECORDS_DIR = GAGENOTES_DIR / "datum_records"

# ---------------------------------------------------------------------------
# Documented assumptions about gage datum handling
# ---------------------------------------------------------------------------
# These assumptions are embedded verbatim into every per-station archive file
# so the rationale for each correction is self-documenting and auditable.
_DATUM_ASSUMPTIONS = """\
# ASSUMPTIONS APPLIED BY LMR_QAQC
# ----------------------------------
# 1. Raw stage values downloaded from the RiverGages API carry no vertical
#    datum metadata. The USACE does not transmit datum information through
#    the API — it is documented only in the gage datasheet PDFs in GageNotes/.
#    The adjustment records in each per-station archive file are therefore the
#    sole authoritative source of datum change history for that station.
#
# 2. Adjustment factors are in FEET and are ADDITIVE:
#      corrected_stage_ft = raw_stage_ft + factor_ft
#    A negative factor lowers stage (e.g. re-levelling found the gage was
#    reading high); a positive factor raises it.
#
# 3. During training only adjustments whose target_datum contains "NAVD88"
#    are applied, normalising all stations to a common NAVD88 reference.
#    Adjustments targeting other datums (e.g. NGVD29, MSL) are archived
#    here for completeness but are NOT applied.
#
# 4. When adjustment periods from different NAVD88 epochs overlap the same
#    calendar dates, the row with the most recent (highest numeric)
#    target_epoch takes priority. Older epoch rows are skipped for those
#    dates to avoid double-counting.
#
# 5. Rows with no end_date are open-ended: the correction is applied from
#    start_date through the end of the available data record.
#
# 6. Rows where (end_date - start_date) < 2 days are excluded. These are
#    single-entry datum-transition bookkeeping records used by the USACE to
#    mark the instant of a re-levelling, not sustained corrections to the
#    continuous stage time series.
"""


class GageNotesParser:
    """
    Reads USACE Gage Datasheet PDFs from GageNotes/ and extracts the
    "Gage Calibration Adjustments" table, which records every documented
    datum shift at each gage.

    Table columns (as found in the PDFs)
    --------------------------------------
    Calibration Vertical Datum | Target Vertical Datum | Adjustment Factor |
    Adjustment Start Date      | Adjustment End Date   | Adjustment Notes

    Rows are grouped under "Target Epoch: <epoch>" section headers.

    Per-station archive files
    -------------------------
    The first time a PDF is parsed, the extracted records are written to a
    human-readable plain-text file in GageNotes/datum_records/.  The file
    contains a full header block documenting all assumptions so the provenance
    of every correction is self-contained and auditable without running code.

    On subsequent runs, parse_all() detects the archive file and loads from it
    instead of re-parsing the PDF.  To force a re-parse (e.g. after updating
    a PDF), delete the corresponding .txt file in datum_records/ or pass
    use_cache=False to parse_all().

    The JSON file cache/qaqc/gagenotes_adjustments.json is a compiled summary
    of all per-station records and is written on every run for convenience.
    """

    def __init__(self, gagenotes_dir: Path = GAGENOTES_DIR):
        self.gagenotes_dir = gagenotes_dir

    # ------------------------------------------------------------------
    # Per-station archive file helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _station_record_path(pdf_path: Path) -> Path:
        """Return the archive .txt path that corresponds to *pdf_path*."""
        return DATUM_RECORDS_DIR / f"{pdf_path.stem}_datum_adjustments.txt"

    @staticmethod
    def _write_station_record(
            record_path: Path,
            pdf_name: str,
            sid: str,
            adj_rows: List[dict],
    ) -> None:
        """
        Write a human-readable pipe-delimited archive file for one station.

        The file header embeds the full assumptions block so it is auditable
        without any supporting code.  Data lines are pipe-delimited with one
        adjustment record per line.  Empty end_date means open-ended.
        """
        record_path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "# ============================================================",
            "# Datum Adjustment Archive — LMR_QAQC System",
            f"# Source PDF  : {pdf_name}",
            f"# USACE Code  : {sid}",
            f"# Generated   : {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')} UTC",
            "#",
        ]
        header += _DATUM_ASSUMPTIONS.splitlines()
        header += [
            "#",
            "# COLUMNS (pipe-delimited; empty end_date = open-ended correction)",
            "# cal_datum|target_datum|factor_ft|start_date|end_date|target_epoch|notes",
            "# ============================================================",
        ]
        data_lines = []
        for row in adj_rows:
            data_lines.append("|".join([
                row.get("cal_datum", ""),
                row.get("target_datum", ""),
                str(row.get("factor_ft", "")),
                row.get("start_date", ""),
                row.get("end_date", "") or "",
                row.get("target_epoch", ""),
                row.get("notes", "").replace("\n", " "),
            ]))
        record_path.write_text(
            "\n".join(header + data_lines) + "\n",
            encoding="utf-8",
        )
        log.info(
            "GageNotes: archived %d adjustment records → %s",
            len(adj_rows), record_path.relative_to(ROOT),
        )

    @staticmethod
    def _read_station_record(record_path: Path) -> Optional[Tuple[str, List[dict]]]:
        """
        Load a per-station archive .txt file.

        Returns (usace_code, adj_rows) on success, or None if the file cannot
        be parsed.  An empty adj_rows list is valid — it means the PDF was
        previously parsed and contained no adjustment records (no datum changes
        documented for that station).
        """
        try:
            text = record_path.read_text(encoding="utf-8")
        except OSError:
            return None

        sid: Optional[str] = None
        adj_rows: List[dict] = []

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                # Extract USACE code from the header comment
                m = re.match(
                    r'#\s*USACE Code\s*:\s*(\d{5})', stripped, re.IGNORECASE)
                if m:
                    sid = m.group(1)
                continue
            # Data line — pipe-delimited
            parts = [p.strip() for p in stripped.split("|")]
            if len(parts) < 4:
                continue
            try:
                factor = float(parts[2].replace("+", "").strip())
            except ValueError:
                continue
            adj_rows.append({
                "cal_datum":    parts[0],
                "target_datum": parts[1] if len(parts) > 1 else "",
                "factor_ft":    factor,
                "start_date":   parts[3] if len(parts) > 3 else "",
                "end_date":     parts[4] if len(parts) > 4 and parts[4] else None,
                "target_epoch": parts[5] if len(parts) > 5 else "",
                "notes":        parts[6] if len(parts) > 6 else "",
            })

        if sid is None:
            return None
        return sid, adj_rows

    # ------------------------------------------------------------------
    # Internal PDF helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_usace_code(text: str) -> Optional[str]:
        """
        Pull the 5-digit USACE station code out of the first-page text.
        Handles suffixes like '01670 OD' or '01100Q' by taking only digits.
        """
        m = re.search(r'USACE\s+Code\s+(\d{5})', text, re.IGNORECASE)
        return m.group(1) if m else None

    @staticmethod
    def _parse_adjustment_table(pages) -> List[dict]:
        """
        Walk all pages extracting rows from the Gage Calibration Adjustments
        table.  Tracks 'Target Epoch' section headers to record which target
        datum epoch each row belongs to.  Rows without a parseable numeric
        factor or a start date are skipped.
        """
        rows: List[dict] = []
        current_epoch = ""

        for page in pages:
            for table in (page.extract_tables() or []):
                in_adj_table = False
                for row in table:
                    cells = [str(c or "").replace("\n", " ").strip()
                             for c in row]
                    header_str = " ".join(cells)

                    # Identify the adjustments table by its header
                    if "Adjustment Factor" in header_str and "Calibration" in header_str:
                        in_adj_table = True
                        current_epoch = ""
                        continue

                    if not in_adj_table:
                        continue

                    # Target Epoch section separator
                    if cells[0].startswith("Target Epoch:"):
                        ep_m = re.search(r'Target Epoch:\s*(.+)', cells[0])
                        current_epoch = ep_m.group(1).strip() if ep_m else ""
                        continue

                    # Need at least 4 cells and a parseable factor
                    if len(cells) < 4 or not cells[2]:
                        continue
                    try:
                        factor = float(cells[2].replace("+", "").strip())
                    except ValueError:
                        continue

                    # Parse start date (required)
                    start_dt = pd.to_datetime(
                        cells[3], errors="coerce") if cells[3] else pd.NaT
                    if pd.isna(start_dt):
                        continue

                    # Parse end date (optional — NaT means open-ended)
                    end_dt = pd.NaT
                    if len(cells) > 4 and cells[4]:
                        end_dt = pd.to_datetime(cells[4], errors="coerce")

                    rows.append({
                        "cal_datum":    cells[0],
                        "target_datum": cells[1] if len(cells) > 1 else "",
                        "factor_ft":    factor,
                        "start_date":   start_dt.isoformat(),
                        "end_date":     end_dt.isoformat() if not pd.isna(end_dt) else None,
                        "target_epoch": current_epoch,
                        "notes":        cells[5] if len(cells) > 5 else "",
                    })

        return rows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def parse_all(self, use_cache: bool = True) -> Dict[str, List[dict]]:
        """
        Load datum adjustment records for every gage datasheet PDF.

        For each PDF the method checks for a pre-existing per-station archive
        file in GageNotes/datum_records/.  If the file exists it is loaded
        directly — the PDF is never opened.  If no archive file exists the PDF
        is parsed, the records are written to a new archive file, and then the
        records are used.  This means each PDF is parsed at most once.

        Parameters
        ----------
        use_cache : bool
            When True (default), use existing per-station archive .txt files.
            When False, re-parse every PDF and overwrite the archive files.
            Use False after replacing a PDF with an updated version.

        Returns
        -------
        dict  station_id (5-digit USACE code) → list of adjustment row dicts.
        """
        try:
            import pdfplumber
        except ImportError:
            log.error(
                "pdfplumber is not installed. Install it with:  pip install pdfplumber\n"
                "Datum adjustment corrections from GageNotes PDFs will be skipped."
            )
            return {}

        result: Dict[str, List[dict]] = {}
        pdfs = sorted(self.gagenotes_dir.glob("*.pdf"))
        if not pdfs:
            log.warning("No PDFs found in %s", self.gagenotes_dir)
            return result

        DATUM_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
        new_parsed = 0
        loaded_from_archive = 0

        prog = ProgressTracker(len(pdfs), "GageNotes")
        for pdf_path in pdfs:
            prog.update(0, item_name=pdf_path.stem[:15])
            record_path = self._station_record_path(pdf_path)

            # ── Try loading from per-station archive file first ────────────
            if use_cache and record_path.exists():
                loaded = self._read_station_record(record_path)
                if loaded is not None:
                    sid, adj_rows = loaded
                    if adj_rows:
                        result.setdefault(sid, []).extend(adj_rows)
                    loaded_from_archive += 1
                    prog.update(1, item_name=pdf_path.stem[:15])
                    continue
                # File corrupt/unreadable — fall through to re-parse
                log.warning(
                    "Archive file unreadable, re-parsing PDF: %s", record_path.name
                )

            # ── Parse the PDF (first time or use_cache=False) ──────────────
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    first_text = pdf.pages[0].extract_text() or ""
                    sid = self._extract_usace_code(first_text)
                    if sid is None:
                        log.debug(
                            "No USACE code found in %s — skipping", pdf_path.name)
                        prog.update(1, item_name=pdf_path.stem[:15])
                        continue
                    adj_rows = self._parse_adjustment_table(pdf.pages)
            except Exception as exc:
                log.warning("Failed to parse %s: %s", pdf_path.name, exc)
                prog.update(1, item_name=pdf_path.stem[:15])
                continue

            # ── Write archive file for future runs ─────────────────────────
            self._write_station_record(
                record_path, pdf_path.name, sid, adj_rows)
            new_parsed += 1

            if adj_rows:
                result.setdefault(sid, []).extend(adj_rows)
            prog.update(1, item_name=pdf_path.stem[:15])
        prog.finish()

        total = sum(len(v) for v in result.values())
        log.info(
            "GageNotes: %d PDFs loaded from archive, %d newly parsed — "
            "%d adjustment records across %d stations",
            loaded_from_archive, new_parsed, total, len(result),
        )

        # Write / refresh the combined JSON summary
        GAGENOTES_ADJ_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with GAGENOTES_ADJ_CACHE.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

        return result

    @staticmethod
    def build_offset_series(
            adj_rows: List[dict],
            date_index: pd.DatetimeIndex,
            target_datum: str = "NAVD88",
            min_span_days: int = 2,
    ) -> pd.Series:
        """
        Build a time-indexed Series of additive offsets (feet) for one station.

        Only rows whose target_datum contains *target_datum* are used.
        Rows with a date span of less than *min_span_days* are skipped because
        single-day entries are datum-transition bookkeeping records rather than
        corrections to apply to the full time series.

        When multiple rows overlap (different target epochs), the row belonging
        to the highest numeric epoch wins (i.e. most modern NAVD88 epoch takes
        priority over older NAVD88 realizations).

        Returns a float Series of offsets; 0.0 where no adjustment applies.
        """
        offsets = pd.Series(0.0, index=date_index, dtype=float)

        applicable = [
            r for r in adj_rows
            if target_datum.upper() in r.get("target_datum", "").upper()
        ]
        if not applicable:
            return offsets

        def _epoch_key(r: dict) -> float:
            try:
                return float(r.get("target_epoch", "0") or "0")
            except (ValueError, TypeError):
                return 0.0

        # Sort most-recent epoch first so it claims dates before older epochs
        applicable_sorted = sorted(applicable, key=_epoch_key, reverse=True)

        assigned = pd.Series(False, index=date_index)

        for row in applicable_sorted:
            start = pd.to_datetime(row["start_date"], errors="coerce")
            end_raw = row.get("end_date")
            end = pd.to_datetime(
                end_raw, errors="coerce") if end_raw else pd.NaT
            factor = float(row["factor_ft"])

            if pd.isna(start):
                continue

            # Skip single-day bookkeeping entries
            if not pd.isna(end):
                span = (end - start).days
                if span < min_span_days:
                    continue
                mask = (date_index >= start) & (
                    date_index <= end) & (~assigned)
            else:
                # Open-ended: applies from start_date through the full record
                mask = (date_index >= start) & (~assigned)

            offsets[mask] = factor
            assigned[mask] = True

        return offsets

    @staticmethod
    def match_station_id(pdf_sid: str, rg_ids: List[str]) -> Optional[str]:
        """
        Match a 5-digit PDF USACE code against the live station ID list.

        RiverGages station IDs sometimes carry a letter suffix (e.g. '01100Q').
        We match on the first 5 digits.
        """
        for sid in rg_ids:
            if re.match(rf'^{re.escape(pdf_sid)}', sid):
                return sid
        return None

    def apply_to_dataframe(
            self,
            rg_df: pd.DataFrame,
            rg_ids: List[str],
            target_datum: str = "NAVD88",
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Apply datum adjustments to all stage columns in *rg_df*.

        Returns
        -------
        corrected_df : copy of rg_df with offsets applied
        applied      : dict  station_id → number of days corrected
        """
        adj_data = self.parse_all()
        if not adj_data:
            return rg_df.copy(), {}

        corrected = rg_df.copy()
        applied: Dict[str, int] = {}

        for pdf_sid, adj_rows in adj_data.items():
            matched_sid = self.match_station_id(pdf_sid, rg_ids)
            if matched_sid is None:
                continue
            col = f"stage_{matched_sid}"
            if col not in corrected.columns:
                continue
            offsets = self.build_offset_series(
                adj_rows, corrected.index, target_datum)
            nonzero = int((offsets != 0.0).sum())
            if nonzero > 0:
                corrected[col] = corrected[col] + offsets
                applied[matched_sid] = nonzero
                log.info(
                    "Datum correction applied to station %s: %d days adjusted (→ %s)",
                    matched_sid, nonzero, target_datum,
                )

        return corrected, applied


# ---------------------------------------------------------------------------
# Vertical datum conversion (NOAA/NGS VDatum REST API)
# ---------------------------------------------------------------------------
class VDatumAPI:
    """
    Thin wrapper around the NOAA/NGS VDatum REST API.
    https://vdatum.noaa.gov/vdatumweb/api/convert

    All elevations are exchanged with the API in **metres**.
    The caller is responsible for unit conversion before/after calling.

    Common vertical datum codes
    ---------------------------
    Orthometric : NAVD88, NGVD29
    Tidal       : LMSL, MLLW, MHHW, MLW, MHW, MTL, DTL
    Ellipsoidal : NAD83_2011, WGS84_G1674

    Tidal datums require coordinates within the VDatum tidal model boundary.
    Inland stations will receive a JSON 'message' key instead of 't_z'.
    """

    def __init__(self):
        self.base_url = VDATUM_BASE

    def convert_elevation(
            self,
            lon: float,
            lat: float,
            height: float = 0.0,
            s_h_frame: str = "NAD83_2011",
            s_v_frame: str = "NAVD88",
            t_h_frame: str = "NAD83_2011",
            t_v_frame: str = "LMSL",
            region: str = "contiguous",
    ) -> Optional[dict]:
        """
        Transform a single elevation from one vertical datum to another.

        Parameters
        ----------
        lon, lat   : Decimal degrees. West longitudes are negative.
        height     : Elevation in **metres** referenced to *s_v_frame*.
        s_v_frame  : Source vertical datum code (e.g. 'NAVD88', 'NGVD29').
        t_v_frame  : Target vertical datum code (e.g. 'LMSL', 'MLLW', 'NAVD88').
        region     : 'contiguous' for CONUS; 'ak', 'hi', 'prvi', etc. otherwise.

        Returns
        -------
        dict with 't_z' (converted elevation in metres) on success.
        dict with 'message' key if the location is outside the tidal model.
        None on network/HTTP error.
        """
        params = {
            "s_x": lon,
            "s_y": lat,
            "s_z": height,
            "s_h_frame": s_h_frame,
            "s_v_frame": s_v_frame,
            "s_v_unit": "m",
            "t_h_frame": t_h_frame,
            "t_v_frame": t_v_frame,
            "t_v_unit": "m",
            "region": region,
        }
        try:
            r = requests.get(self.base_url, params=params,
                             timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as exc:
            log.warning("VDatum API error at (%.4f, %.4f): %s", lat, lon, exc)
            return None


class DatumConverter:
    """
    Per-station vertical datum offset manager.

    Workflow
    --------
    1. For each station, query VDatum with height=0 m.  The returned t_z is
       the additive offset between the two datums at that lat/lon.
    2. Cache offsets in *VDATUM_OFFSET_CACHE* so subsequent runs are instant.
    3. All stage inputs/outputs are in **feet**; metres are used only
       internally when talking to the VDatum API.

    Usage
    -----
    converter = DatumConverter()
    series_lmsl = converter.convert_series(series_navd88, sid, lat, lon,
                                            s_v_frame="NAVD88", t_v_frame="LMSL")

    convert_network() handles an entire stage DataFrame at once.
    """

    def __init__(self):
        self._cache: Dict[str, float] = {}
        self._api = VDatumAPI()
        self._load_cache()

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------
    @staticmethod
    def _key(sid: str, s_v_frame: str, t_v_frame: str) -> str:
        return f"{sid}::{s_v_frame}::{t_v_frame}"

    def _load_cache(self) -> None:
        if VDATUM_OFFSET_CACHE.exists():
            try:
                with VDATUM_OFFSET_CACHE.open("r", encoding="utf-8") as fh:
                    self._cache = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning(
                    "VDatum offset cache unreadable, starting fresh: %s", exc)
                self._cache = {}

    def _save_cache(self) -> None:
        VDATUM_OFFSET_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with VDATUM_OFFSET_CACHE.open("w", encoding="utf-8") as fh:
            json.dump(self._cache, fh, indent=2)

    # ------------------------------------------------------------------
    # Offset resolution
    # ------------------------------------------------------------------
    def get_offset_ft(
            self,
            sid: str,
            lat: float,
            lon: float,
            s_v_frame: str,
            t_v_frame: str,
            region: str = "contiguous",
    ) -> Optional[float]:
        """
        Return the additive offset (feet) so that::

            stage_converted_ft = stage_ft + offset_ft

        The offset equals t_z from VDatum when queried with s_z=0, converted
        to feet.  Returns None when VDatum cannot resolve the location (e.g.
        an orthometric-to-tidal conversion at an inland coordinate).
        """
        if s_v_frame == t_v_frame:
            return 0.0

        key = self._key(sid, s_v_frame, t_v_frame)
        if key in self._cache:
            return float(self._cache[key])

        result = self._api.convert_elevation(
            lon=lon,
            lat=lat,
            height=0.0,
            s_v_frame=s_v_frame,
            t_v_frame=t_v_frame,
            region=region,
        )

        if result is None:
            return None

        if "t_z" not in result:
            msg = result.get("message", "no 't_z' in response")
            log.warning(
                "VDatum cannot convert %s→%s at station %s (%.4f, %.4f): %s",
                s_v_frame, t_v_frame, sid, lat, lon, msg,
            )
            return None

        # t_z for s_z=0 is the datum separation
        offset_m = float(result["t_z"])
        offset_ft = offset_m * M_TO_FT
        self._cache[key] = offset_ft
        self._save_cache()
        log.info(
            "VDatum offset %s→%s at station %s (%.4f, %.4f): %+.4f ft",
            s_v_frame, t_v_frame, sid, lat, lon, offset_ft,
        )
        return offset_ft

    # ------------------------------------------------------------------
    # Series and DataFrame conversion
    # ------------------------------------------------------------------
    def convert_series(
            self,
            stage_ft: pd.Series,
            sid: str,
            lat: float,
            lon: float,
            s_v_frame: str,
            t_v_frame: str,
            region: str = "contiguous",
    ) -> pd.Series:
        """
        Apply datum conversion to a stage series (feet in, feet out).

        If the offset cannot be resolved the original series is returned
        unchanged and a warning is logged.
        """
        if s_v_frame == t_v_frame:
            return stage_ft.copy()

        offset_ft = self.get_offset_ft(
            sid, lat, lon, s_v_frame, t_v_frame, region)
        if offset_ft is None:
            log.warning(
                "Datum conversion skipped for station %s (%s→%s): "
                "location may be outside the tidal model boundary.",
                sid, s_v_frame, t_v_frame,
            )
            return stage_ft.copy()

        return stage_ft + offset_ft

    def convert_network(
            self,
            stage_df: pd.DataFrame,
            station_coords: Dict[str, Dict[str, float]],
            s_v_frame: str,
            t_v_frame: str,
            region: str = "contiguous",
    ) -> pd.DataFrame:
        """
        Convert all stage columns in *stage_df* to *t_v_frame*.

        Parameters
        ----------
        stage_df       : DataFrame with 'stage_<sid>' columns in feet.
        station_coords : Mapping  sid → {'lat': float, 'lon': float}.
        s_v_frame      : Source vertical datum (applies to all stations).
        t_v_frame      : Target vertical datum.

        Returns
        -------
        Copy of *stage_df* with converted values.  Stations with unknown
        coordinates are left unconverted.
        """
        if s_v_frame == t_v_frame:
            return stage_df.copy()

        stage_cols = [c for c in stage_df.columns if c.startswith("stage_")]
        out = stage_df.copy()
        prog = ProgressTracker(len(stage_cols), "Datum Convert")
        for col in stage_cols:
            sid = col.replace("stage_", "")
            coords = station_coords.get(sid, {})
            prog.update(0, item_name=sid)
            if "lat" not in coords or "lon" not in coords:
                log.warning(
                    "No coordinates for station %s — datum conversion skipped.", sid)
                prog.update(1, item_name=sid)
                continue
            out[col] = self.convert_series(
                stage_df[col], sid, coords["lat"], coords["lon"],
                s_v_frame, t_v_frame, region,
            )
            prog.update(1, item_name=sid)
        prog.finish()
        return out


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------
def load_rg_station_ids() -> List[str]:
    if not RG_STATIONS_FILE.exists():
        raise FileNotFoundError(f"Missing station file: {RG_STATIONS_FILE}")

    lines = [x.strip() for x in RG_STATIONS_FILE.read_text(
        encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return []
    if lines[0].lower().startswith("stations"):
        lines = lines[1:]
    return lines


def load_usgs_station_ids() -> List[str]:
    if not USGS_STATIONS_FILE.exists():
        return []
    df = pd.read_csv(USGS_STATIONS_FILE)
    col = "USGS_Station_Number" if "USGS_Station_Number" in df.columns else df.columns[1]
    # Ensure USGS stations are properly 8-digit zero-padded
    return [str(x).strip().zfill(8) for x in df[col].dropna().tolist()]


def inventory_gagenotes() -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    if not GAGENOTES_DIR.exists():
        return result

    for p in GAGENOTES_DIR.rglob("*.pdf"):
        sid_match = re.findall(r"(\d{5})", p.stem)
        sid = sid_match[0] if sid_match else "unmatched"
        result.setdefault(sid, []).append(str(p.relative_to(ROOT)))
    return result


# ---------------------------------------------------------------------------
# Travel-time calibration: peak/trough matching + cross-correlation
# ---------------------------------------------------------------------------
def find_peaks_troughs(
        series: pd.Series,
        half_window: int = PEAK_HALF_WINDOW_DAYS,
        min_prom_sigma: float = PEAK_MIN_PROM_SIGMA,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Locate significant peaks and troughs in a daily stage series.

    Uses a rolling-window local-extremum test combined with a prominence filter
    so that minor ripples do not generate spurious matches.  The prominence
    threshold is expressed as a fraction of the robust σ of the series so it
    scales automatically with the amplitude of each gauge.

    Returns (peaks, troughs) as DatetimeIndex objects.
    """
    s = series.dropna()
    if len(s) < 2 * half_window + 1:
        return pd.DatetimeIndex([]), pd.DatetimeIndex([])

    sigma = robust_sigma(s.values.astype(float))
    min_prom = max(min_prom_sigma * sigma, 0.01)  # at least 0.01 ft

    vals = s.values.astype(float)
    idx = s.index
    n = len(vals)
    peaks, troughs = [], []

    for i in range(half_window, n - half_window):
        win = vals[i - half_window: i + half_window + 1]
        local_max = float(np.nanmax(win))
        local_min = float(np.nanmin(win))
        if local_max - local_min < min_prom:
            continue
        # Peak: centre must strictly exceed all neighbours within window
        neighbours = np.concatenate(
            [vals[i - half_window: i].astype(float), vals[i + 1: i + half_window + 1].astype(float)])
        if not np.isnan(vals[i]) and vals[i] >= local_max and float(vals[i]) > float(np.nanmax(neighbours)):
            peaks.append(idx[i])
        elif not np.isnan(vals[i]) and vals[i] <= local_min and float(vals[i]) < float(np.nanmin(neighbours)):
            troughs.append(idx[i])

    return pd.DatetimeIndex(peaks), pd.DatetimeIndex(troughs)


def xcorr_lag(
        s_up: pd.Series,
        s_dn: pd.Series,
        max_lag: int = MAX_XCORR_LAG_DAYS,
) -> int:
    """
    Find the integer lag L (days) that maximises the cross-correlation between
    the upstream series shifted forward by L days and the downstream series.

    Before correlating, a 90-day centred rolling mean is subtracted so the
    correlation reflects flood-event timing rather than the slow seasonal signal.
    This is important on the LMR where the seasonal cycle is much larger in
    amplitude than individual flood events.
    """
    common = s_up.index.intersection(s_dn.index)
    if len(common) < 90:
        return 0

    a = s_up.reindex(common).interpolate(limit=5)
    b = s_dn.reindex(common).interpolate(limit=5)

    # High-pass: remove slow seasonal baseline
    rolling_kw = dict(window=90, min_periods=30, center=True)
    a = (a - a.rolling(**rolling_kw).mean()).fillna(0).values
    b = (b - b.rolling(**rolling_kw).mean()).fillna(0).values

    best_lag, best_corr = 0, -np.inf
    for lag in range(0, max_lag + 1):
        if lag == 0:
            corr = float(np.corrcoef(a, b)[0, 1])
        else:
            # upstream at t-lag → downstream at t  (upstream leads)
            corr = float(np.corrcoef(a[:-lag], b[lag:])[0, 1])
        if corr > best_corr:
            best_corr, best_lag = corr, lag

    return best_lag


def calibrate_pair_lags(
        s_up: pd.Series,
        s_dn: pd.Series,
        q_series: Optional[pd.Series] = None,
        max_lag: int = PEAK_MATCH_MAX_DAYS,
        half_window: int = PEAK_HALF_WINDOW_DAYS,
) -> Dict[str, int]:
    """
    Estimate travel-time lag (days) from upstream→downstream by matching
    identifiable peaks and troughs in the stage records.

    Algorithm
    ---------
    1. Detect peaks and troughs in both series.
    2. For each upstream extremum, find the nearest downstream extremum
       within the search window [t_up, t_up + max_lag].
    3. Record the matched lag and the concurrent discharge at that time.
    4. Bin matched lags by discharge quartile to produce discharge-
       conditional lag estimates:  high Q → shorter lag,  low Q → longer lag.
    5. Fall back to cross-correlation when fewer than 5 peaks are matched.

    Returns a dict with keys: 'overall', 'low', 'medium_low', 'medium_high',
    'high' — all integer lag days.
    """
    # Cross-correlation estimate as an unconditional baseline
    xc = xcorr_lag(s_up, s_dn, max_lag=max_lag)
    result: Dict[str, int] = {"overall": xc}
    for label in DISCHARGE_BIN_LABELS:
        result[label] = xc

    # Peak/trough matching
    up_peaks, up_troughs = find_peaks_troughs(s_up, half_window=half_window)
    dn_peaks, dn_troughs = find_peaks_troughs(s_dn, half_window=half_window)

    extrema_up = pd.DatetimeIndex(sorted(list(up_peaks) + list(up_troughs)))
    extrema_dn = pd.DatetimeIndex(sorted(list(dn_peaks) + list(dn_troughs)))

    if len(extrema_up) < 5 or len(extrema_dn) < 5:
        log.debug("  Peak matching: too few extrema (%d up, %d dn) — using xcorr=%d",
                  len(extrema_up), len(extrema_dn), xc)
        return result

    matched_lags: List[float] = []
    matched_q: List[float] = []
    last_matched_dn = pd.Timestamp.min  # avoid duplicate downstream matches

    for t_up in extrema_up:
        win_dn = extrema_dn[
            (extrema_dn >= t_up) &
            (extrema_dn <= t_up + pd.Timedelta(days=max_lag)) &
            (extrema_dn > last_matched_dn)
        ]
        if len(win_dn) == 0:
            continue
        t_dn = win_dn[0]  # earliest available downstream extremum
        lag_days = (t_dn - t_up).days
        if lag_days < 0:
            continue
        matched_lags.append(float(lag_days))
        last_matched_dn = t_dn

        # Concurrent discharge at upstream event time (nearest valid day)
        if q_series is not None:
            q_window = q_series.loc[
                t_up - pd.Timedelta(days=1): t_up + pd.Timedelta(days=1)
            ].dropna()
            matched_q.append(float(q_window.mean())
                             if len(q_window) > 0 else np.nan)
        else:
            matched_q.append(np.nan)

    if len(matched_lags) < 5:
        log.debug("  Peak matching: only %d pairs — using xcorr=%d",
                  len(matched_lags), xc)
        return result

    lag_arr = np.array(matched_lags)
    q_arr = np.array(matched_q)

    # Overall median from peak matching (replace xcorr estimate)
    result["overall"] = int(round(float(np.nanmedian(lag_arr))))

    # Discharge-conditional medians (high Q → faster travel → shorter lag)
    q_valid_mask = ~np.isnan(q_arr)
    if q_series is not None and q_valid_mask.sum() >= 8:
        q_cuts = np.nanpercentile(q_arr[q_valid_mask], [0, 25, 50, 75, 100])
        for i, label in enumerate(DISCHARGE_BIN_LABELS):
            mask = (q_arr >= q_cuts[i]) & (
                q_arr <= q_cuts[i + 1]) & q_valid_mask
            if mask.sum() >= 3:
                result[label] = int(round(float(np.nanmedian(lag_arr[mask]))))

    log.info(
        "  Lag calibration: %d matched pairs | overall=%d d | "
        "low=%d d | high=%d d",
        len(matched_lags),
        result["overall"],
        result.get("low", -1),
        result.get("high", -1),
    )
    return result


def calibrate_network_lags(
        rg_df: pd.DataFrame,
        usgs_df: pd.DataFrame,
        rg_ids: List[str],
        rg_miles: Dict[str, float],
) -> Dict[str, Dict[str, int]]:
    """
    Calibrate travel-time lags for all meaningful upstream→downstream station
    pairs in the RiverGages network.

    Only pairs with:
      - Both stations having known river miles
      - Separation ≤ 600 river miles
      - At least 3 intermediate stations between them (i+1 … i+3 in ordered list)
    are calibrated.  Longer-range pairs are less useful because backwater
    effects and tributary inflows break the simple translation model.

    Results are keyed as "<upstream_sid>_to_<downstream_sid>" (i.e., higher
    river mile → lower river mile on the LMR).
    """
    # Sort stations from upstream (high RM) to downstream (low RM = Head of Passes)
    ordered = sorted(
        [(sid, mile) for sid, mile in rg_miles.items()
         if f"stage_{sid}" in rg_df.columns],
        key=lambda x: x[1],
        reverse=True,  # highest RM first
    )

    if len(ordered) < 2:
        log.warning(
            "Fewer than 2 stations with river miles — skipping lag calibration.")
        return {}

    # Pick the USGS discharge series with the best data coverage as the
    # conditioning signal for discharge-conditional lags.
    q_cond: Optional[pd.Series] = None
    best_cov = 0
    for col in usgs_df.columns:
        cov = int(usgs_df[col].notna().sum())
        if cov > best_cov:
            best_cov, q_cond = cov, usgs_df[col]

    lag_table: Dict[str, Dict[str, int]] = {}

    for i, (up_sid, up_mile) in enumerate(ordered):
        s_up = rg_df[f"stage_{up_sid}"].dropna()
        # Calibrate against up to 3 nearest downstream neighbours only
        for j in range(i + 1, min(i + 4, len(ordered))):
            dn_sid, dn_mile = ordered[j]
            dist = up_mile - dn_mile
            if dist > 600:
                continue
            s_dn = rg_df[f"stage_{dn_sid}"].dropna()
            pair_key = f"{up_sid}_to_{dn_sid}"
            log.info(
                "Calibrating lag: %s (RM %.0f) → %s (RM %.0f)  [%.0f mi]",
                up_sid, up_mile, dn_sid, dn_mile, dist,
            )
            try:
                lag_info = calibrate_pair_lags(s_up, s_dn, q_series=q_cond)
                lag_table[pair_key] = lag_info
            except Exception as exc:
                log.warning("Lag calibration failed for %s: %s", pair_key, exc)

    log.info("Lag calibration complete: %d station pairs", len(lag_table))
    return lag_table


# ---------------------------------------------------------------------------
# Feature engineering and modeling
# ---------------------------------------------------------------------------
def build_base_frame(
        rg_stages_daily: pd.DataFrame,
        usgs_discharge_daily: pd.DataFrame,
        noaa_daily: pd.DataFrame,
) -> pd.DataFrame:
    df = rg_stages_daily.copy()

    for c in usgs_discharge_daily.columns:
        df[c] = usgs_discharge_daily[c]

    for c in noaa_daily.columns:
        df[c] = noaa_daily[c]

    if {"wl_8761724", "wl_8729840"}.issubset(set(df.columns)):
        df["gulf_mean"] = df[["wl_8761724", "wl_8729840"]].mean(axis=1)
        df["gulf_spread"] = df["wl_8761724"] - df["wl_8729840"]

    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    return df.sort_index()


def make_station_feature_table(
        base_df: pd.DataFrame,
        target_station: str,
        rg_station_ids: List[str],
        rg_miles: Optional[Dict[str, float]] = None,
        lag_table: Optional[Dict[str, Dict[str, int]]] = None,
) -> pd.DataFrame:
    """
    Build the feature matrix for training a QAQC model at *target_station*.

    New columns are accumulated in a plain dict and assembled with a single
    pd.concat call to avoid the PerformanceWarning caused by repeated
    DataFrame column assignment.

    HistGradientBoostingRegressor handles NaN natively, so no row is dropped
    here for missing feature values — only rows where the target itself is
    missing are excluded (done inside fit_iterative_qaqc).
    """
    target_col = f"stage_{target_station}"
    if target_col not in base_df.columns:
        raise ValueError(f"Target station column missing: {target_col}")

    # All new feature Series are collected here; one concat at the end.
    new_cols: Dict[str, pd.Series] = {}
    feature_cols: List[str] = []
    idx = base_df.index
    target_mile = (rg_miles or {}).get(target_station)
    target_series = base_df[target_col]
    
    # Detect dataframe resolution to scale lags accurately
    if len(base_df) > 1:
        delta = (base_df.index[1] - base_df.index[0]).total_seconds()
        steps_per_day = 24 if delta <= 3600 else 1
    else:
        steps_per_day = 1

    # ── Autoregressive features (hysteresis proxy) ─────────────────────────
    for lag in [1, 2, 3, 5, 7]:
        lag_steps = lag * steps_per_day
        col_name = f"{target_col}_lag{lag}d"
        new_cols[col_name] = target_series.shift(lag_steps)
        feature_cols.append(col_name)

    _d1 = target_series.diff(1 * steps_per_day)
    _d3 = target_series.diff(3 * steps_per_day)
    _d1r7 = _d1.rolling(7 * steps_per_day, min_periods=3 * steps_per_day).mean()
    new_cols[f"{target_col}_d1"] = _d1
    new_cols[f"{target_col}_d3"] = _d3
    new_cols[f"{target_col}_d1_roll7"] = _d1r7
    feature_cols.extend([
        f"{target_col}_d1",
        f"{target_col}_d3",
        f"{target_col}_d1_roll7",
    ])

    # ── Composite discharge proxy (dimensionless, median-normalised) ────────
    q_cols = [c for c in base_df.columns if c.startswith("q_")]
    if q_cols:
        q_composite = pd.concat(
            [base_df[c] / max(float(base_df[c].median()), 1.0)
             for c in q_cols],
            axis=1,
        ).mean(axis=1)
    else:
        q_composite = pd.Series(1.0, index=idx)

    # ── Cross-station network features with discharge-adaptive lags ─────────
    for sid in rg_station_ids:
        if sid == target_station:
            continue
        col = f"stage_{sid}"
        if col not in base_df.columns:
            continue

        neighbor_series = base_df[col]
        base_name = f"net_{sid}"
        neighbor_mile = (rg_miles or {}).get(sid)
        dist = (
            abs(neighbor_mile - target_mile)
            if (neighbor_mile is not None and target_mile is not None)
            else None
        )

        # Determine lag window centre: calibrated table first, physics fallback
        lag_center: int = 0
        lag_low_q: int = 0
        lag_high_q: int = 2

        cal: Dict[str, int] = {}
        if lag_table and neighbor_mile is not None and target_mile is not None:
            pair_key = (
                f"{sid}_to_{target_station}"
                if neighbor_mile > target_mile
                else f"{target_station}_to_{sid}"
            )
            cal = lag_table.get(pair_key, {})

        if cal:
            lag_center = int(cal.get("overall", 0))
            lag_low_q = int(cal.get("high", lag_center))  # high Q → faster
            lag_high_q = int(cal.get("low", lag_center))  # low Q  → slower
        elif dist is not None and dist > 0:
            lag_center = max(0, int(round(dist / FLOW_SPEED_BASE_MI_PER_DAY)))
            lag_low_q = max(
                0, int(round(dist / (FLOW_SPEED_BASE_MI_PER_DAY * 1.4))))
            lag_high_q = max(
                0, int(round(dist / (FLOW_SPEED_BASE_MI_PER_DAY * 0.6))))

        # Lagged stage features spanning the calibrated lag window
        lags_to_use = sorted(set([
            max(0, lag_low_q),
            max(0, lag_center - 1),
            max(0, lag_center),
            max(0, lag_center + 1),
            max(0, lag_high_q),
        ]))
        for lag in lags_to_use:
            lag_steps = lag * steps_per_day
            feat = f"{base_name}_lag{lag}d"
            if feat not in new_cols:
                new_cols[feat] = neighbor_series.shift(lag_steps)
                feature_cols.append(feat)

        # Rate-of-change at the central lag
        d1_feat = f"{base_name}_d1_lag{lag_center}d"
        if d1_feat not in new_cols:
            new_cols[d1_feat] = neighbor_series.diff(1 * steps_per_day).shift(lag_center * steps_per_day)
            feature_cols.append(d1_feat)

        # Per-day physically expected travel time (discharge-adaptive)
        if dist is not None and dist > 0:
            q_safe = q_composite.clip(lower=0.05)
            speed_series = FLOW_SPEED_BASE_MI_PER_DAY * \
                (q_safe ** FLOW_SPEED_EXPONENT)
            exp_lag_feat = f"expected_lag_{sid}_days"
            if exp_lag_feat not in new_cols:
                new_cols[exp_lag_feat] = (
                    dist / speed_series).clip(lower=0, upper=30).round()
                feature_cols.append(exp_lag_feat)

    # ── Local Spatial Gradient Features ────────────────────────────────────
    # Identify immediate upstream and downstream neighbors to explicitly
    # calculate the local water surface slope.
    if target_mile is not None and rg_miles:
        ups = [(s, m) for s, m in rg_miles.items() if m > target_mile and f"stage_{s}" in base_df.columns]
        dns = [(s, m) for s, m in rg_miles.items() if m < target_mile and f"stage_{s}" in base_df.columns]
        
        if ups:
            up_sid = min(ups, key=lambda x: x[1] - target_mile)[0]
            for lag in [1, 2, 3, 5]:
                lag_steps = lag * steps_per_day
                feat = f"local_grad_up_lag{lag}d"
                new_cols[feat] = base_df[f"stage_{up_sid}"].shift(lag_steps) - target_series.shift(lag_steps)
                feature_cols.append(feat)
                
        if dns:
            dn_sid = min(dns, key=lambda x: target_mile - x[1])[0]
            for lag in [1, 2, 3, 5]:
                lag_steps = lag * steps_per_day
                feat = f"local_grad_dn_lag{lag}d"
                new_cols[feat] = target_series.shift(lag_steps) - base_df[f"stage_{dn_sid}"].shift(lag_steps)
                feature_cols.append(feat)

    # ── USGS discharge features (lagged) ───────────────────────────────────
    for c in q_cols:
        for lag in [0, 1, 2, 3, 5, 7]:
            lag_steps = lag * steps_per_day
            qcol = f"{c}_lag{lag}d"
            if qcol not in new_cols:
                new_cols[qcol] = base_df[c].shift(lag_steps)
                feature_cols.append(qcol)

    # ── Gulf / tidal base-level + composite discharge features ─────────────
    for c in ["wl_8761724", "wl_8729840", "gulf_mean", "gulf_spread", "doy_sin", "doy_cos"]:
        if c in base_df.columns:
            for lag in [0, 1, 2, 3]:
                lag_steps = lag * steps_per_day
                wcol = f"{c}_lag{lag}d"
                if wcol not in new_cols:
                    new_cols[wcol] = base_df[c].shift(lag_steps)
                    feature_cols.append(wcol)

    for lag in [0, 1, 2, 3]:
        lag_steps = lag * steps_per_day
        wcol = f"q_composite_lag{lag}d"
        if wcol not in new_cols:
            new_cols[wcol] = q_composite.shift(lag_steps)
            feature_cols.append(wcol)

    # ── Assemble in a single concat (avoids DataFrame fragmentation) ────────
    feature_df = pd.DataFrame(new_cols, index=idx)
    table = pd.concat([feature_df, base_df[[target_col]]], axis=1)
    table = table[feature_cols + [target_col]
                  ].replace([np.inf, -np.inf], np.nan)
    return table


def fit_iterative_qaqc(
        table: pd.DataFrame,
        target_col: str,
        z_thresh: float = OUTLIER_Z,
        max_iter: int = ITER_MAX,
) -> Tuple[StationModelBundle, pd.DataFrame]:
    feature_columns = [c for c in table.columns if c != target_col]

    # HistGradientBoostingRegressor supports NaN in features natively, so we
    # only exclude rows where the TARGET observation is itself missing.
    usable = table[table[target_col].notna()].copy()
    if usable.empty:
        raise ValueError(
            "No trainable rows: target column has no valid observations")

    train_mask = pd.Series(True, index=usable.index)
    all_outliers = pd.Series(False, index=usable.index)
    pred_all = pd.Series(np.nan, index=usable.index)
    model: Optional[HistGradientBoostingRegressor] = None

    for _ in range(max_iter):
        fit_df = usable.loc[train_mask].copy()
        if len(fit_df) < 120:
            break

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=8,
            max_iter=500,
            learning_rate=0.04,
            min_samples_leaf=20,
            l2_regularization=0.05,
            random_state=42,
        )
        model.fit(fit_df[feature_columns], fit_df[target_col])

        pred_all = pd.Series(model.predict(
            usable[feature_columns]), index=usable.index)
        resid = usable[target_col] - pred_all
        sigma = robust_sigma(resid.values.astype(float))
        if sigma <= 0 or np.isnan(sigma):
            break

        current_outliers = resid.abs() > (z_thresh * sigma)
        prev_count = int(all_outliers.sum())
        all_outliers = all_outliers | current_outliers
        train_mask = ~all_outliers
        if int(all_outliers.sum()) == prev_count:
            break

    if model is None:
        raise ValueError("Model training failed")

    pred_final = pd.Series(model.predict(usable[feature_columns]), index=usable.index)
    pred_all_imputed = pd.Series(model.predict(table[feature_columns]), index=table.index)
    obs_final = usable[target_col]
    mae = float(mean_absolute_error(obs_final, pred_final))
    rmse = float(mean_squared_error(obs_final, pred_final) ** 0.5)
    sigma = robust_sigma((obs_final - pred_final).values.astype(float))
    flagged = (obs_final - pred_final).abs() > (z_thresh * sigma)

    bundle = StationModelBundle(
        station_id=target_col.replace("stage_", ""),
        feature_columns=feature_columns,
        model=model,
        metrics={
            "rows_total": float(len(usable)),
            "rows_train_final": float((~flagged).sum()),
            "mae": mae,
            "rmse": rmse,
            "sigma": float(sigma),
            "flagged_count": float(flagged.sum()),
            "flagged_fraction": float(flagged.mean()),
        },
        outlier_timestamps=[x.isoformat() for x in usable.index[flagged]],
    )

    result = pd.DataFrame(index=table.index)
    result["observed"] = table[target_col]
    result["predicted"] = pred_all_imputed
    result["residual"] = table[target_col] - pred_all_imputed
    flagged_all = pd.Series(False, index=table.index)
    flagged_all.loc[usable.index] = flagged
    result["is_outlier"] = flagged_all
    result["corrected"] = np.where(flagged_all | table[target_col].isna(), pred_all_imputed, table[target_col])
    return bundle, result


# ---------------------------------------------------------------------------
# Gap-fill spatial validation
# ---------------------------------------------------------------------------

def _validate_and_repair_gapfills(
        corrected_network: pd.DataFrame,
        raw_network: pd.DataFrame,
        rg_miles: Dict[str, float],
        bundles: Dict[str, "StationModelBundle"],
        spike_threshold_ft: float = GAPFILL_SPATIAL_THRESHOLD_FT,
        z_thresh: float = OUTLIER_Z,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Secondary validation pass for gap-fill predictions.

    For every station/date where the raw observation is NaN the corrected value
    equals the model's prediction.  Two independent checks are applied:

    **Fix 1 — Spatial cross-check**
        The predicted value is compared to a linear interpolation between the
        nearest upstream and downstream neighbours that have valid (non-gap-fill)
        corrected values on the same date.  When the deviation exceeds
        ``spike_threshold_ft`` the prediction is replaced by the spatial estimate
        and the date is recorded in the returned ``gapfill_suspect`` mask.

    **Fix 2 — Prediction confidence interval**
        Each station's training residual sigma is used to form a ±(z × σ) band
        around the original model prediction.  When the spatial estimate lies
        *outside* this band it independently confirms that the model is
        extrapolating implausibly; a WARNING is logged even if the spatial
        deviation is below ``spike_threshold_ft``.

    Returns
    -------
    corrected : pd.DataFrame
        Updated corrected_network with suspect gap-fills replaced by spatial
        interpolation.
    gapfill_suspect : pd.DataFrame
        Boolean mask (same shape) — True where a gap-fill was repaired.
    ci_lower : pd.DataFrame
        Lower bound of model prediction CI (only populated for gap-fill rows).
    ci_upper : pd.DataFrame
        Upper bound of model prediction CI (only populated for gap-fill rows).
    """
    # Stations ordered upstream → downstream by river mile.
    stations_with_miles: List[Tuple[str, float]] = sorted(
        [(sid, mile) for sid, mile in rg_miles.items()
         if f"stage_{sid}" in corrected_network.columns],
        key=lambda x: x[1],
        reverse=True,
    )

    # Per-station robust sigma from training (floor 0.25 ft).
    sigma_map: Dict[str, float] = {
        sid: max(b.metrics.get("sigma", 0.5), 0.25)
        for sid, b in bundles.items()
    }

    corrected = corrected_network.copy()
    gapfill_suspect = pd.DataFrame(
        False, index=corrected.index, columns=corrected.columns)
    ci_lower = pd.DataFrame(
        np.nan, index=corrected.index, columns=corrected.columns)
    ci_upper = pd.DataFrame(
        np.nan, index=corrected.index, columns=corrected.columns)

    # Populate CI bounds for all gap-fill rows using original (pre-repair) predictions.
    for col in corrected_network.columns:
        sid = col.replace("stage_", "")
        if col not in raw_network.columns:
            continue
        is_gapfill = raw_network[col].isna().reindex(corrected.index).fillna(False)
        if not is_gapfill.any():
            continue
        ci_half = z_thresh * sigma_map.get(sid, 0.5)
        ci_lower.loc[is_gapfill, col] = corrected_network.loc[is_gapfill, col] - ci_half
        ci_upper.loc[is_gapfill, col] = corrected_network.loc[is_gapfill, col] + ci_half

    # Spatial cross-check and repair.
    for ts_idx, (sid, rm_i) in enumerate(stations_with_miles):
        col = f"stage_{sid}"
        upstream_stations   = stations_with_miles[:ts_idx]    # higher RM
        downstream_stations = stations_with_miles[ts_idx + 1:]  # lower RM

        # Edge gages (no upstream+downstream pair) are exempt — same logic as
        # LMR_surfer's _check_monotonicity().
        # LMR_surfer's _check_monotonicity().
        if not upstream_stations or not downstream_stations:
            continue

        if col not in raw_network.columns:
            continue
        is_gapfill = raw_network[col].isna().reindex(corrected.index).fillna(False)
        if not is_gapfill.any():
            continue

        sid_up, rm_up = upstream_stations[-1]
        sid_dn, rm_dn = downstream_stations[0]
        col_up = f"stage_{sid_up}"
        col_dn = f"stage_{sid_dn}"

        span = rm_up - rm_dn
        if span <= 0:
            continue
        frac = (rm_up - rm_i) / span

        val_up = corrected[col_up]
        val_dn = corrected[col_dn]
        both_valid = val_up.notna() & val_dn.notna()

        # For REPAIR only use dates where the immediate neighbours have actual
        # raw observations (not themselves gap-fills).  This prevents a cascade
        # where a bad gap-fill anchor is used to "correct" adjacent stations.
        up_has_raw = (
            raw_network[col_up].notna().reindex(corrected.index, fill_value=False)
            if col_up in raw_network.columns
            else pd.Series(False, index=corrected.index)
        )
        dn_has_raw = (
            raw_network[col_dn].notna().reindex(corrected.index, fill_value=False)
            if col_dn in raw_network.columns
            else pd.Series(False, index=corrected.index)
        )
        both_raw = up_has_raw & dn_has_raw   # strict: repair only with real obs anchors

        # Linear interpolation by river mile.
        spatial_est = val_up + (val_dn - val_up) * frac

        # Use original (pre-repair) model prediction for deviation measurement.
        predicted_val = corrected_network[col]
        deviation = (predicted_val - spatial_est).abs()

        sigma_i = sigma_map.get(sid, 0.5)
        ci_half = z_thresh * sigma_i

        # Fix 1: spatial deviation exceeds hard threshold → repair.
        # Restricted to both_raw dates to avoid cascade errors.
        to_repair = is_gapfill & both_raw & (deviation > spike_threshold_ft)

        # Fix 2: spatial estimate lies outside model PI → CI warning only.
        # Uses both_valid (allows gap-fill anchors) for informational loggingeal obs anchors

        # Linear interpolation by river mile.
        spatial_est = val_up + (val_dn - val_up) * frac

        # Use original (pre-repair) model prediction for deviation measurement.
        predicted_val = corrected_network[col]
        deviation = (predicted_val - spatial_est).abs()

        sigma_i = sigma_map.get(sid, 0.5)
        ci_half = z_thresh * sigma_i

        # Fix 1: spatial deviation exceeds hard threshold → repair.
        # Restricted to both_raw dates to avoid cascade errors.
        to_repair = is_gapfill & both_raw & (deviation > spike_threshold_ft)

        # Fix 2: spatial estimate lies outside model PI → CI warning only.
        # Uses both_valid (allows gap-fill anchors) for informational logging.
        ci_violated = is_gapfill & both_valid & (deviation > ci_half) & ~to_repair

        if to_repair.any():
            n = int(to_repair.sum())
            gapfill_suspect[col] = gapfill_suspect[col] | to_repair
            corrected.loc[to_repair, col] = spatial_est.loc[to_repair]
            log.warning(
                "Gap-fill REPAIRED %s: %d date(s) — model predicted %.1f ft "
                "but spatial profile suggests %.1f ft "
                "(max Δ=%.1f ft, threshold=%.1f ft) → replaced with spatial interpolation",
                sid, n,
                float(predicted_val[to_repair].iloc[0]) if n == 1 else float(
                    predicted_val[to_repair].mean()),
                float(spatial_est[to_repair].iloc[0]) if n == 1 else float(
                    spatial_est[to_repair].mean()),
                float(deviation[to_repair].max()),
                spike_threshold_ft,
            )

        if ci_violated.any():
            n_ci = int(ci_violated.sum())
            log.warning(
                "Gap-fill CI WARNING %s: %d date(s) — spatial estimate lies "
                "outside model prediction interval (±%.1f ft), "
                "max Δ=%.1f ft — below %.1f ft spatial repair threshold, "
                "prediction retained but flagged",
                sid, n_ci, ci_half,
                float(deviation[ci_violated].max()),
                spike_threshold_ft,
            )
            # Mark in suspect mask with a lower-confidence flag (value = 0.5 as float
            # would conflict; instead we keep gapfill_suspect boolean and rely on logs).

    total_repaired = int(gapfill_suspect.to_numpy().sum())
    log.info(
        "Gap-fill validation complete: %d station-day(s) repaired via spatial interpolation",
        total_repaired,
    )
    return corrected, gapfill_suspect, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------
def collect_historical_data(start: pd.Timestamp, end: pd.Timestamp, freq: str = DEFAULT_FREQ, rg_miles: Dict[str, float] = None) -> Dict[str, pd.DataFrame]:
    if rg_miles is None:
        rg_miles = {}
    log.info("--- MILESTONE 1: Collecting Historical Data ---")
    rg_ids = load_rg_station_ids()
    usgs_ids = load_usgs_station_ids()
    
    global_freq = "h" if freq in ("H", "h", "hourly", "hybrid") else "D"

    rg_client = RiverGagesClient()

    rg_series: Dict[str, pd.Series] = {}
    prog_rg = ProgressTracker(len(rg_ids), "RiverGages Stage")
    for sid in rg_ids:
        prog_rg.update(0, item_name=sid)
        if freq == "hybrid":
            mile = rg_miles.get(sid, 0)
            s_freq = "h" if mile < 228.0 else "D"
        elif freq in ("H", "h", "hourly"):
            s_freq = "h"
        else:
            s_freq = "D"
            
        s = rg_client.fetch_stage(
            sid, start, end, variable=TARGET_STAGE_VARIABLE)
        if not s.empty:
            s = s.sort_index().resample(s_freq).mean()
            if s_freq == "D" and global_freq == "h":
                s = s.resample("h").ffill()
            rg_series[f"stage_{sid}"] = s
        prog_rg.update(1, item_name=sid)
    prog_rg.finish()

    usgs_client = USGSClient()
    usgs_series: Dict[str, pd.Series] = {}
    prog_usgs = ProgressTracker(len(usgs_ids), "USGS Discharge")
    for sid in usgs_ids:
        prog_usgs.update(0, item_name=sid)
        if freq == "hybrid" or freq in ("D", "d", "daily"):
            s_freq = "D"
        else:
            s_freq = "h"
        try:
            s = usgs_client.fetch_discharge(sid, start, end, freq=s_freq)
            if not s.empty:
                s = s.sort_index().resample(s_freq).mean()
                if s_freq == "D" and global_freq == "h":
                    s = s.resample("h").ffill()
                usgs_series[f"q_{sid}"] = s
        except Exception as exc:
            log.warning("\nUSGS fetch failed for %s: %s", sid, exc)
        prog_usgs.update(1, item_name=sid)
    prog_usgs.finish()

    noaa_client = NOAAClient()
    noaa_series: Dict[str, pd.Series] = {}
    prog_noaa = ProgressTracker(len(NOAA_STATIONS), "NOAA Water Level")
    for label, sid in NOAA_STATIONS.items():
        prog_noaa.update(0, item_name=label)
        if freq == "hybrid" or freq in ("H", "h", "hourly"):
            s_freq = "h"
        else:
            s_freq = "D"
        try:
            s = noaa_client.fetch_water_level(sid, start, end, freq=s_freq)
            if not s.empty:
                s = s.sort_index().resample(s_freq).mean()
                if s_freq == "D" and global_freq == "h":
                    s = s.resample("h").ffill()
                noaa_series[f"wl_{sid}"] = s
        except Exception as exc:
            log.warning("\nNOAA fetch failed for %s (%s): %s", label, sid, exc)
        prog_noaa.update(1, item_name=label)
    prog_noaa.finish()

    idx = pd.date_range(start=start.normalize(), 
                        end=end.normalize() + pd.Timedelta(hours=23 if global_freq=="h" else 0), freq=global_freq)

    rg_df = pd.DataFrame(index=idx)
    for c, s in rg_series.items():
        rg_df[c] = s.reindex(idx)

    usgs_df = pd.DataFrame(index=idx)
    for c, s in usgs_series.items():
        usgs_df[c] = s.reindex(idx)

    noaa_df = pd.DataFrame(index=idx)
    for c, s in noaa_series.items():
        noaa_df[c] = s.reindex(idx)

    return {
        "rg": rg_df,
        "usgs": usgs_df,
        "noaa": noaa_df,
    }


def run_training(
        years: int = DEFAULT_LOOKBACK_YEARS,
        freq: str = DEFAULT_FREQ,
        z_thresh: float = OUTLIER_Z,
        max_iter: int = ITER_MAX,
) -> None:
    ensure_dirs()

    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = (end - pd.DateOffset(years=years)).normalize()

    log.info("--- MILESTONE 1a: Mapping Stations ---")
    rg_ids = load_rg_station_ids()
    usgs_ids = load_usgs_station_ids()
    mapper = RiverMileMapper()
    rg_client = RiverGagesClient()
    rg_miles: Dict[str, float] = {}
    rg_coords: Dict[str, Dict[str, float]] = {}
    prog_miles = ProgressTracker(len(rg_ids), "Mapping Stations")
    for sid in rg_ids:
        prog_miles.update(0, item_name=sid)
        meta = rg_client.get_station_metadata(sid)
        if meta:
            if "lat" in meta and "lon" in meta:
                rg_coords[sid] = {"lat": meta["lat"], "lon": meta["lon"]}
            if not mapper.data.empty:
                mile = mapper.coords_to_mile(meta["lat"], meta["lon"])
                if not np.isnan(mile):
                    rg_miles[sid] = mile
        prog_miles.update(1, item_name=sid)
    prog_miles.finish()

    log.info("Training window: %s to %s", start.date(), end.date())
    history = collect_historical_data(start, end, freq=freq, rg_miles=rg_miles)

    rg_df = history["rg"]
    usgs_df = history["usgs"]
    noaa_df = history["noaa"]

    if rg_df.empty:
        raise RuntimeError("No RiverGages data collected. Cannot train QAQC model.")

    # ── Apply datum adjustments from GageNotes PDFs ─────────────────────────
    # This normalises each station's raw stage to a single consistent vertical
    # datum (NAVD88) before any feature engineering or model training.
    # Corrections are sourced from the "Gage Calibration Adjustments" tables
    # parsed directly from the PDF datasheets in GageNotes/.
    log.info("--- MILESTONE 1b: Applying GageNotes Datum Corrections ---")
    gagenotes_parser = GageNotesParser()
    rg_df, datum_applied = gagenotes_parser.apply_to_dataframe(rg_df, rg_ids)
    if datum_applied:
        log.info(
            "Datum corrections applied to %d station(s): %s",
            len(datum_applied),
            ", ".join(f"{s}({d}d)" for s, d in sorted(
                datum_applied.items())),
        )
    else:
        log.info("No datum corrections found in GageNotes for current station set.")

    base = build_base_frame(rg_df, usgs_df, noaa_df)

    # ── Calibrate inter-station travel-time lags from peak matching ─────────
    # This builds a discharge-conditional lag lookup table used in feature
    # engineering.  Flood-wave speed increases with discharge (power-law V∝Q^0.4)
    # so lags are shorter at high flow and longer at low flow.
    lag_table_path = OUTPUT_DIR / "lag_table.json"
    if lag_table_path.exists():
        log.info("--- MILESTONE 3: Loaded Calibrated Travel Times from Cache ---")
        with lag_table_path.open("r", encoding="utf-8") as _lf:
            lag_table = json.load(_lf)
    else:
        log.info("--- MILESTONE 3: Calibrating Network Travel Times ---")

        # Re-route the internal looping log prints by redefining calibrate_network_lags' loops
        pairs = []
        ordered = sorted([(sid, mile) for sid, mile in rg_miles.items(
        ) if f"stage_{sid}" in rg_df.columns], key=lambda x: x[1], reverse=True)
        for i, (up_sid, up_mile) in enumerate(ordered):
            for j in range(i + 1, min(i + 4, len(ordered))):
                dn_sid, dn_mile = ordered[j]
                if up_mile - dn_mile <= 600:
                    pairs.append((up_sid, dn_sid))

        prog_lags = ProgressTracker(len(pairs), "Calibrating Lags")
        lag_table = {}
        
        rg_df_calib = rg_df.resample('D').mean()
        usgs_df_calib = usgs_df.resample('D').mean()
        q_cond = next(
            (usgs_df_calib[c] for c in usgs_df_calib.columns if usgs_df_calib[c].notna().sum() > 0), None)
        for up_sid, dn_sid in pairs:
            pair_key = f"{up_sid}_to_{dn_sid}"
            prog_lags.update(0, item_name=pair_key)
            try:
                lag_info = calibrate_pair_lags(rg_df_calib[f"stage_{up_sid}"].dropna(), rg_df_calib[f"stage_{dn_sid}"].dropna(), q_series=q_cond)
                lag_table[pair_key] = lag_info
            except Exception as exc:
                log.warning("\nLag calibration failed for %s: %s",
                            pair_key, exc)
            prog_lags.update(1, item_name=pair_key)
        prog_lags.finish()

        with lag_table_path.open("w", encoding="utf-8") as _lf:
            json.dump(lag_table, _lf, indent=2)
        log.info("Lag table saved to %s", lag_table_path)

    checkpoint_path = CACHE_DIR / "training_checkpoint.pkl"
    bundles: Dict[str, StationModelBundle] = {}
    metrics_rows: List[Dict[str, float]] = []
    if checkpoint_path.exists():
        log.info(
            "--- MILESTONE 4: Resuming Iterative Model Training from Checkpoint ---")
        with open(checkpoint_path, "rb") as f:
            chk = pickle.load(f)
            bundles = chk["bundles"]
            corrected_network = chk["corrected"]
            predicted_network = chk["predicted"]
            outlier_network = chk["outlier"]
            metrics_rows = chk["metrics"]
    else:
        log.info("--- MILESTONE 4: Iterative Model Training ---")
        corrected_network = pd.DataFrame(index=base.index)
        predicted_network = pd.DataFrame(index=base.index)
        outlier_network = pd.DataFrame(index=base.index)

    prog_train = ProgressTracker(len(rg_ids), "Training Models")
    if bundles:
        prog_train.update(len(bundles), item_name="Resuming...")

    for sid in rg_ids:
        if sid in bundles:
            continue

        target_col = f"stage_{sid}"
        if target_col not in base.columns:
            log.warning("\nSkipping station %s: no data available in base frame", sid)
            prog_train.update(1, item_name=sid)
            continue

        prog_train.update(0, item_name=sid)
        table = make_station_feature_table(
            base, sid, rg_ids, rg_miles, lag_table)

        try:
            bundle, result = fit_iterative_qaqc(
                table=table,
                target_col=target_col,
                z_thresh=z_thresh,
                max_iter=max_iter,
            )
        except Exception as exc:
            log.warning(
                "\nSkipping station %s due to model failure: %s", sid, exc)
            prog_train.update(1, item_name=sid)
            continue

        bundles[sid] = bundle
        corrected_network[f"stage_{sid}"] = result["corrected"].reindex(
            base.index)
        predicted_network[f"stage_{sid}"] = result["predicted"].reindex(
            base.index)
        outlier_network[f"stage_{sid}"] = result["is_outlier"].reindex(
            base.index).fillna(False)

        row = {"station_id": sid}
        row.update(bundle.metrics)
        metrics_rows.append(row)

        result_out = OUTPUT_DIR / \
            f"station_{sanitize_station_id(sid)}_qaqc_timeseries.csv"
        result.reset_index(names="date").to_csv(result_out, index=False)

        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "bundles": bundles,
                "corrected": corrected_network,
                "predicted": predicted_network,
                "outlier": outlier_network,
                "metrics": metrics_rows
            }, f)

        prog_train.update(1, item_name=sid)
    prog_train.finish()

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    if not bundles:
        raise RuntimeError("No station models were successfully trained.")

    # ── Fix 1 + Fix 2: validate gap-fill predictions against the spatial network
    # and model prediction confidence intervals.  Bad gap-fills are replaced in
    # corrected_network before any outputs are written.
    log.info("--- MILESTONE 4b: Validating Gap-Fill Spatial Consistency ---")
    corrected_network, gapfill_suspect_network, gapfill_ci_lower, gapfill_ci_upper = \
        _validate_and_repair_gapfills(
            corrected_network=corrected_network,
            raw_network=rg_df.reindex(base.index),
            rg_miles=rg_miles,
            bundles=bundles,
            spike_threshold_ft=GAPFILL_SPATIAL_THRESHOLD_FT,
            z_thresh=z_thresh,
        )

    notes_inventory = inventory_gagenotes()

    log.info("--- MILESTONE 5: Generating Output Artifacts ---")
    # Persist all assets needed for reproducibility and future querying.
    artifact = {
        "created_utc": datetime.utcnow().isoformat(),
        "train_start": start.isoformat(),
        "train_end": end.isoformat(),
        "frequency": freq,
        "params": {
            "years": years,
            "z_thresh": z_thresh,
            "max_iter": max_iter,
        },
        "rg_station_ids": rg_ids,
        "rg_miles": rg_miles,
        "rg_coords": rg_coords,
        "lag_table": lag_table,
        "model_bundles": bundles,
        "gagenotes_pdf_inventory": notes_inventory,
        "datum_corrections_applied": datum_applied,
    }

    artifact_path = MODEL_DIR / "network_qaqc_artifact.pkl"
    with artifact_path.open("wb") as f:
        pickle.dump(artifact, f)

    rg_df.reset_index(names="date").to_csv(
        OUTPUT_DIR / "raw_stage_network.csv", index=False)
    corrected_network.reset_index(names="date").to_csv(
        OUTPUT_DIR / "corrected_stage_network.csv", index=False)
    predicted_network.reset_index(names="date").to_csv(
        OUTPUT_DIR / "predicted_stage_network.csv", index=False)
    outlier_network.reset_index(names="date").to_csv(
        OUTPUT_DIR / "outlier_flags_network.csv", index=False)
    gapfill_suspect_network.reset_index(names="date").to_csv(
        OUTPUT_DIR / "gapfill_suspect_network.csv", index=False)
    gapfill_ci_lower.reset_index(names="date").to_csv(
        OUTPUT_DIR / "gapfill_ci_lower_network.csv", index=False)
    gapfill_ci_upper.reset_index(names="date").to_csv(
        OUTPUT_DIR / "gapfill_ci_upper_network.csv", index=False)

    pd.DataFrame(metrics_rows).to_csv(
        OUTPUT_DIR / "model_metrics.csv", index=False)

    metadata = {
        "artifact_path": str(artifact_path.relative_to(ROOT)),
        "outputs_dir": str(OUTPUT_DIR.relative_to(ROOT)),
        "station_count": len(bundles),
        "train_start": start.isoformat(),
        "train_end": end.isoformat(),
        "frequency": freq,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata,
                                                             indent=2), encoding="utf-8")

    log.info("Training complete. Trained stations: %d", len(bundles))
    log.info("Artifacts saved: %s", artifact_path)


# ---------------------------------------------------------------------------
# Dashboard/query layer
# ---------------------------------------------------------------------------
def load_trained_outputs() -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifact_path = MODEL_DIR / "network_qaqc_artifact.pkl"
    if not artifact_path.exists():
        raise FileNotFoundError("No artifact found. Run training first.")

    with artifact_path.open("rb") as f:
        artifact = pickle.load(f)

    corrected = pd.read_csv(
        OUTPUT_DIR / "corrected_stage_network.csv", parse_dates=["date"]).set_index("date")
    predicted = pd.read_csv(
        OUTPUT_DIR / "predicted_stage_network.csv", parse_dates=["date"]).set_index("date")
    raw = pd.read_csv(OUTPUT_DIR / "raw_stage_network.csv",
                      parse_dates=["date"]).set_index("date")
    return artifact, raw, corrected, predicted


def query_stage_series(
        target: str,
        start: str,
        end: str,
        source: str = "corrected",
        s_v_frame: str = DEFAULT_SOURCE_DATUM,
        t_v_frame: str = "",
) -> pd.DataFrame:
    """
    Retrieve a stage series for a station ID or LMR River Mile.

    Parameters
    ----------
    target     : RiverGages station ID (e.g. '01120') or a River Mile as a
                 floating-point string (e.g. '306.3').
    start, end : Date range in 'YYYY-MM-DD' format.
    source     : One of 'corrected', 'predicted', 'raw'.
    s_v_frame  : Vertical datum of the stored stage data (default NAVD88).
    t_v_frame  : Target vertical datum for the output.  Leave blank or pass
                 the same value as *s_v_frame* to skip conversion.
                 Examples: 'LMSL', 'MLLW', 'MHHW', 'NGVD29'.

    Returns
    -------
    DataFrame with a 'stage_ft' column indexed by date.  If datum conversion
    was applied, the column reflects the converted datum.
    """
    artifact, raw, corrected, predicted = load_trained_outputs()
    frame_map = {
        "raw": raw,
        "corrected": corrected,
        "predicted": predicted,
    }
    if source not in frame_map:
        raise ValueError("source must be one of: raw, corrected, predicted")

    df = frame_map[source]
    s = parse_iso(start)
    e = parse_iso(end)
    window = df.loc[(df.index >= s) & (df.index <= e)].copy()

    col = f"stage_{target}"
    # station whose coords guide datum conversion
    _ref_sid: Optional[str] = None

    if col in window.columns:
        out = window[[col]].rename(columns={col: "stage_ft"})
        _ref_sid = target
    else:
        try:
            target_mile = float(target)
        except ValueError:
            available = ", ".join(
                sorted([c.replace("stage_", "") for c in df.columns if c.startswith("stage_")]))
            raise KeyError(
                f"'{target}' is not a valid station ID or River Mile. Available stations: {available}")

        rg_miles = artifact.get("rg_miles", {})
        if not rg_miles:
            raise ValueError(
                "No River Mile mappings found in artifact. Cannot interpolate.")

        stations_with_miles = [(sid, mile) for sid, mile in rg_miles.items(
        ) if f"stage_{sid}" in window.columns]
        stations_with_miles.sort(key=lambda x: x[1])

        below = [x for x in stations_with_miles if x[1] <= target_mile]
        above = [x for x in stations_with_miles if x[1] >= target_mile]

        if not below and not above:
            raise ValueError("No valid stations for interpolation.")

        if not below:
            s_col = f"stage_{above[0][0]}"
            out = window[[s_col]].rename(columns={s_col: "stage_ft"})
            _ref_sid = above[0][0]
        elif not above:
            s_col = f"stage_{below[-1][0]}"
            out = window[[s_col]].rename(columns={s_col: "stage_ft"})
            _ref_sid = below[-1][0]
        else:
            lo_sid, lo_mile = below[-1]
            hi_sid, hi_mile = above[0]

            if lo_sid == hi_sid:
                s_col = f"stage_{lo_sid}"
                out = window[[s_col]].rename(columns={s_col: "stage_ft"})
                _ref_sid = lo_sid
            else:
                span = hi_mile - lo_mile
                frac = (target_mile - lo_mile) / span if span > 0 else 0.5
                interp_series = window[f"stage_{lo_sid}"] * \
                    (1 - frac) + window[f"stage_{hi_sid}"] * frac
                out = pd.DataFrame({"stage_ft": interp_series})
                _ref_sid = lo_sid  # use lower-bound station for datum reference
                log.info(
                    "Interpolated RM %.1f between %s (RM %.1f) and %s (RM %.1f)",
                    target_mile, lo_sid, lo_mile, hi_sid, hi_mile,
                )

    # --- Vertical datum conversion ----------------------------------------
    effective_t = t_v_frame.strip().upper() if t_v_frame else ""
    effective_s = s_v_frame.strip().upper() if s_v_frame else DEFAULT_SOURCE_DATUM.upper()
    if effective_t and effective_t != effective_s and _ref_sid is not None:
        rg_coords = artifact.get("rg_coords", {})
        coords = rg_coords.get(_ref_sid, {})
        if coords and "lat" in coords and "lon" in coords:
            converter = DatumConverter()
            out["stage_ft"] = converter.convert_series(
                out["stage_ft"],
                _ref_sid,
                coords["lat"],
                coords["lon"],
                s_v_frame=effective_s,
                t_v_frame=effective_t,
            )
            log.info("Output datum: %s (converted from %s)",
                     effective_t, effective_s)
        else:
            log.warning(
                "No coordinates stored for station %s — datum conversion skipped. "
                "Re-run 'train' to populate station coordinates.",
                _ref_sid,
            )

    return out


def run_dashboard() -> None:
    artifact, _, _, _ = load_trained_outputs()
    ids = artifact.get("rg_station_ids", [])
    print("LMR QAQC Dashboard")
    print("Available stations:", ", ".join(ids))
    print(f"Default source datum : {DEFAULT_SOURCE_DATUM}")
    print("Common target datums : LMSL  MLLW  MHHW  MLW  MHW  NGVD29  NAVD88")
    print("Type 'exit' to quit")

    while True:
        target = input("Target (Station ID or River Mile): ").strip()
        if target.lower() in {"exit", "quit"}:
            break
        start = input("Start date (YYYY-MM-DD): ").strip()
        end = input("End date (YYYY-MM-DD): ").strip()
        source = input(
            "Source [corrected/predicted/raw] (default corrected): ").strip() or "corrected"
        s_datum = input(f"Source datum (default {DEFAULT_SOURCE_DATUM}): ").strip(
        ) or DEFAULT_SOURCE_DATUM
        t_datum = input(
            "Target datum (leave blank to keep source datum): ").strip()
        try:
            out = query_stage_series(
                target, start, end, source=source,
                s_v_frame=s_datum, t_v_frame=t_datum,
            )
            print(out.head(10))
            print(f"Rows: {len(out)}")
            if t_datum and t_datum.upper() != s_datum.upper():
                print(
                    f"Vertical datum: {t_datum.upper()} (converted from {s_datum.upper()})")
        except Exception as exc:
            print(f"Query failed: {exc}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LMR stage QAQC + correction model builder and query dashboard"
    )
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser(
        "train", help="Train QAQC model and generate corrected outputs")
    train.add_argument("--years", type=int, default=DEFAULT_LOOKBACK_YEARS,
                       help="Historical lookback window")
    train.add_argument("--freq", type=str, default=DEFAULT_FREQ,
                       choices=["D", "d", "daily", "H", "h", "hourly", "hybrid"],
                       help=f"Resampling frequency (default {DEFAULT_FREQ})")
    train.add_argument("--z-thresh", type=float, default=OUTLIER_Z,
                       help="Outlier residual threshold in robust sigma")
    train.add_argument("--max-iter", type=int, default=ITER_MAX,
                       help="Max iterative outlier-removal rounds")

    query = sub.add_parser(
        "query", help="Query trained series for station/date interval")
    query.add_argument("--target", required=True,
                       help="RiverGages station ID or LMR River Mile")
    query.add_argument("--start", required=True, help="YYYY-MM-DD")
    query.add_argument("--end", required=True, help="YYYY-MM-DD")
    query.add_argument("--source", default="corrected",
                       choices=["raw", "corrected", "predicted"])
    query.add_argument("--s-datum", default=DEFAULT_SOURCE_DATUM,
                       help=f"Source vertical datum of stored stage data (default: {DEFAULT_SOURCE_DATUM})")
    query.add_argument("--t-datum", default="",
                       help="Target vertical datum for output "
                            "(e.g. LMSL, MLLW, MHHW, NGVD29). "
                            "Omit to keep the source datum.")
    query.add_argument("--out", default="", help="Optional CSV output path")

    sub.add_parser("dashboard", help="Interactive query dashboard")

    return parser


def main() -> None:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_training(
            years=args.years,
            freq=args.freq,
            z_thresh=args.z_thresh,
            max_iter=args.max_iter,
        )
        return

    if args.command == "query":
        out = query_stage_series(
            target=args.target,
            start=args.start,
            end=args.end,
            source=args.source,
            s_v_frame=args.s_datum,
            t_v_frame=args.t_datum,
        )
        print(out.head(30))
        print(f"Rows returned: {len(out)}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            out.reset_index(names="date").to_csv(args.out, index=False)
            print(f"Saved query output to: {args.out}")
        return

    if args.command == "dashboard":
        run_dashboard()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
