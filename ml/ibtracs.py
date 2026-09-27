"""IBTrACS North Indian Ocean loading and per-observation feature building.

Winds: `wind` is the JTWC 1-min sustained wind (USA_WIND, kt) used for modelling
(longest consistent record, same convention as the TCIR image labels and the RI
literature). `wind_imd` is the RSMC New Delhi 3-min wind (NEWDELHI_WIND) kept for
display and IMD-class comparisons.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "ibtracs", "ibtracs.NI.list.v04r01.csv")
PROCESSED = os.path.join(ROOT, "data", "processed", "ibtracs_ni.parquet")

COLS = {"SID": "sid", "NAME": "name", "SEASON": "season", "SUBBASIN": "subbasin", "ISO_TIME": "time",
        "NATURE": "nature", "LAT": "lat", "LON": "lon", "USA_WIND": "wind", "USA_PRES": "pres",
        "NEWDELHI_WIND": "wind_imd", "NEWDELHI_PRES": "pres_imd", "NEWDELHI_GRADE": "grade_imd",
        "DIST2LAND": "dist2land", "LANDFALL": "landfall", "STORM_SPEED": "speed", "STORM_DIR": "heading"}
LEADS_H = [6, 12, 24, 48, 72]
KM_PER_DEG = 111.2


def load_ibtracs(min_season=1982, refresh=False):
    if os.path.exists(PROCESSED) and not refresh:
        return pd.read_parquet(PROCESSED)
    d = pd.read_csv(RAW, skiprows=[1], low_memory=False, keep_default_na=False, na_values=[" ", ""],
                    usecols=list(COLS) + ["TRACK_TYPE"])
    d = d[d.TRACK_TYPE == "main"].drop(columns="TRACK_TYPE").rename(columns=COLS)
    d["time"] = pd.to_datetime(d.time)
    d = d[(d.season >= min_season) & (d.time.dt.minute == 0) & (d.time.dt.hour % 6 == 0)]
    for c in ["lat", "lon", "wind", "pres", "wind_imd", "pres_imd", "dist2land", "landfall", "speed", "heading"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["name"] = d.name.fillna("UNNAMED")
    d = d.sort_values(["sid", "time"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(PROCESSED), exist_ok=True)
    d.to_parquet(PROCESSED, index=False)
    return d


def _at_offset(d, hours, cols):
    """Values of `cols` for the same storm at time t + hours (NaN if not observed)."""
    s = d[["sid", "time"] + cols].copy()
    s["time"] = s.time - pd.Timedelta(hours=hours)
    suffix = f"_{'p' if hours > 0 else 'm'}{abs(hours)}"
    return d[["sid", "time"]].merge(s, on=["sid", "time"], how="left")[cols].add_suffix(suffix).set_index(d.index)


def displacement_km(lat0, lon0, lat1, lon1):
    """East/north displacement (km) on a local tangent plane."""
    dy = (lat1 - lat0) * KM_PER_DEG
    dx = (lon1 - lon0) * KM_PER_DEG * np.cos(np.radians((lat0 + lat1) / 2))
    return dx, dy


def haversine_km(lat0, lon0, lat1, lon1):
    p0, p1 = np.radians(lat0), np.radians(lat1)
    a = np.sin((p1 - p0) / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(np.radians(lon1 - lon0) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_features(d):
    """Past-only predictors and future targets for every 6-hourly observation."""
    d = d.copy()
    parts = [d]
    for h in [-6, -12, -18, -24]:
        parts.append(_at_offset(d, h, ["wind", "lat", "lon"]))
    for h in LEADS_H:
        parts.append(_at_offset(d, h, ["wind", "lat", "lon", "wind_imd"]))
    f = pd.concat(parts, axis=1)
    f["dw6"] = f.wind - f.wind_m6
    f["dw12"] = f.wind - f.wind_m12
    f["dw24"] = f.wind - f.wind_m24
    mx, my = displacement_km(f.lat_m12, f.lon_m12, f.lat, f.lon)
    f["u_motion"], f["v_motion"] = mx / 12, my / 12  # km/h, past 12 h
    f["month_sin"] = np.sin(2 * np.pi * f.time.dt.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * f.time.dt.month / 12)
    f["is_arabian_sea"] = (f.subbasin == "AS").astype(int)
    first = f.groupby("sid").time.transform("min")
    f["age_h"] = (f.time - first).dt.total_seconds() / 3600
    f["max_wind_so_far"] = f.groupby("sid").wind.cummax()
    f["log_dist2land"] = np.log1p(f.dist2land.clip(lower=0))
    f["dw24_future"] = f.wind_p24 - f.wind
    return f


def split_by_season(season):
    return np.where(season <= 2016, "train", np.where(season <= 2019, "val", "test"))
