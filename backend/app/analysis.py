"""Per-storm, per-time analysis using only information available at that time."""
import numpy as np
import pandas as pd
from shapely.geometry import Point, mapping
from shapely.ops import unary_union

from ml.cyclone_scale import IMD_CODES, WIND_1MIN_TO_IMD, imd_class_index
from ml.ibtracs import KM_PER_DEG, LEADS_H, haversine_km
from ml.train_track import PAST

from .inference import (genesis_model, imd_class_info, pressure_from_wind, ri_model, storm_table,
                        track_forecast)

CS_KT = 34


def storms_summary(min_season=1990):
    f = storm_table()
    f = f[f.season >= min_season]
    g = f.groupby("sid").agg(name=("name", "first"), season=("season", "first"), subbasin=("subbasin", "first"),
                             start=("time", "min"), end=("time", "max"), peak_kt=("wind", "max"),
                             peak_imd_kt=("wind_imd_eff", "max"), ri=("dw24_future", lambda s: bool((s >= 30).any())))
    g = g[g.peak_imd_kt >= CS_KT].reset_index()
    g["peak_class"] = [IMD_CODES[int(i)] for i in imd_class_index(g.peak_imd_kt.fillna(0))]
    return g.sort_values("start", ascending=False)


def storm_rows(sid):
    f = storm_table()
    s = f[f.sid == sid].sort_values("time")
    if s.empty:
        raise KeyError(sid)
    return s


def _forecastable(s):
    """The track model needs 18 h of past positions AND winds (as in training); otherwise use persistence."""
    cols = [f"{v}_m{-h}" for h in PAST if h for v in ("lat", "lon", "wind")] + ["wind"]
    return s[cols].notna().all(1)


def track_payload(sid):
    s = storm_rows(sid)
    has_past = _forecastable(s)
    pts = [{"time": t.isoformat(), "lat": float(la), "lon": float(lo), "wind_kt": _f(w), "wind_imd_kt": _f(wi),
            "imd_class": IMD_CODES[int(imd_class_index(wi if np.isfinite(wi) else 0))], "forecastable": bool(ok)}
           for t, la, lo, w, wi, ok in zip(s.time, s.lat, s.lon, s.wind, s.wind_imd_eff, has_past)]
    return {"sid": sid, "name": s.name.iloc[0], "season": int(s.season.iloc[0]), "subbasin": s.subbasin.iloc[0],
            "points": pts}


def _f(v):
    return None if v is None or not np.isfinite(v) else float(v)


def cone_polygon(lat0, lon0, leads):
    """Union of hulls of consecutive uncertainty circles, in a local km projection."""
    coslat = np.cos(np.radians(lat0))

    def to_xy(la, lo):
        return (lo - lon0) * KM_PER_DEG * coslat, (la - lat0) * KM_PER_DEG

    circles = [Point(0, 0).buffer(15, 32)]
    for ld in leads:
        x, y = to_xy(ld["lat"], ld["lon"])
        circles.append(Point(x, y).buffer(max(ld["cone_radius_km"], 15), 32))
    poly = unary_union([unary_union([a, b]).convex_hull for a, b in zip(circles[:-1], circles[1:])])
    geo = mapping(poly)
    ring = geo["coordinates"][0] if geo["type"] == "Polygon" else max(geo["coordinates"], key=len)[0]
    return [[lon0 + x / (KM_PER_DEG * coslat), lat0 + y / KM_PER_DEG] for x, y in ring]


def _interp_wind(sid, t):
    """IMD-scale wind at time t, interpolated from the storm's reported fixes (25 kt if none exist)."""
    s = storm_rows(sid).set_index("time").wind_imd_eff
    s = s[~s.index.duplicated()]
    if s.notna().sum() == 0:
        return 25.0
    v = s.interpolate(method="time", limit_direction="both").get(t)
    return float(v) if v is not None and np.isfinite(v) else float(s.dropna().iloc[0])


def select_row(sid, time=None):
    """Feature row for a storm at (nearest to) `time`, and whether it has enough history to forecast."""
    s = storm_rows(sid)
    ok = _forecastable(s)
    if time is None:
        cand = s[ok]
        row = cand.iloc[len(cand) // 2:len(cand) // 2 + 1] if len(cand) else s.iloc[:1]
    else:
        idx = (s.time - pd.Timestamp(time)).abs().idxmin()
        row = s.loc[[idx]]
    return row, bool(ok.loc[row.index[0]])


def analysis(sid, time=None):
    row, forecastable = select_row(sid, time)
    r = row.iloc[0]
    wind = float(r.wind) if np.isfinite(r.wind) else float(r.wind_imd_eff / WIND_1MIN_TO_IMD)
    wind_imd = float(r.wind_imd_eff) if np.isfinite(r.wind_imd_eff) else wind * WIND_1MIN_TO_IMD
    estimated = not np.isfinite(wind_imd)
    if estimated:  # no agency reported a wind at this fix: interpolate in time within the storm
        wind_imd = _interp_wind(sid, r.time)
        wind = wind_imd / WIND_1MIN_TO_IMD
    out = {"sid": sid, "name": r["name"], "time": r.time.isoformat(),
           "current": {"lat": float(r.lat), "lon": float(r.lon), "wind_kt": wind, "wind_imd_kt": wind_imd,
                       "wind_estimated": bool(estimated),
                       "pressure_hpa": _f(r.pres_imd) or _f(r.pres) or pressure_from_wind(wind_imd),
                       "imd_class": imd_class_info(wind_imd), "dist2land_km": _f(r.dist2land),
                       "motion_kmh": _f(np.hypot(r.u_motion, r.v_motion)),
                       "heading_deg": _f((np.degrees(np.arctan2(r.u_motion, r.v_motion)) + 360) % 360),
                       "trend_24h_kt": _f(r.dw24)}}

    # track + intensity forecast with calibrated cone
    if forecastable:
        leads = track_forecast(row)[0]
        out["forecast"] = {"method": "LSTM + MC-dropout (calibrated 67% cone)", "leads": leads,
                           "cone": cone_polygon(float(r.lat), float(r.lon), leads)}
    else:
        leads = _persistence(r)
        why = "no JTWC wind reported at this time" if not np.isfinite(r.wind) else "less than 18 h of track history"
        out["forecast"] = {"method": f"persistence fallback ({why})", "leads": leads,
                           "cone": cone_polygon(float(r.lat), float(r.lon), leads)}

    # rapid intensification
    rm = ri_model()
    p = float(rm.prob(row)[0])
    out["ri"] = {"probability": p, "threshold": rm.threshold, "climatology": rm.base_rate,
                 "ratio_to_climatology": p / rm.base_rate, "flag": p >= rm.threshold,
                 "definition": "wind increase >= 30 kt in 24 h", "drivers": rm.explain(row)}

    # genesis: only meaningful while below cyclonic-storm strength
    gm = genesis_model()
    # genesis model was trained on young (<= 48 h), sub-CS systems over the sea; outside that it does not apply
    below_cs = wind_imd < CS_KT
    over_sea = np.isfinite(r.dist2land) and r.dist2land > 0
    young = np.isfinite(r.age_h) and r.age_h <= 48
    if not below_cs:
        reason = "already at Cyclonic Storm strength or above"
    elif not over_sea:
        reason = "centre is over land"
    elif not young:
        reason = "system is older than 48 h (outside the genesis stage)"
    else:
        reason = None
    below_cs = reason is None
    out["genesis"] = {"applicable": bool(below_cs), "reason": reason,
                      "probability": float(gm.prob(row)[0]) if below_cs else None,
                      "definition": "becomes Cyclonic Storm (IMD >= 34 kt) within 72 h",
                      "drivers": gm.explain(row) if below_cs else []}

    # what actually happened (backtest)
    actual = []
    for h in LEADS_H:
        la, lo, w = r.get(f"lat_p{h}"), r.get(f"lon_p{h}"), r.get(f"wind_p{h}")
        fc = next(x for x in leads if x["lead_h"] == h)
        if la is not None and np.isfinite(la):
            err = float(haversine_km(fc["lat"], fc["lon"], la, lo))
            actual.append({"lead_h": h, "lat": float(la), "lon": float(lo), "wind_kt": _f(w), "track_error_km": err,
                           "inside_cone": err <= fc["cone_radius_km"],
                           "wind_error_kt": _f(fc["wind_kt"] - w) if w is not None and np.isfinite(w) else None})
    out["actual"] = actual
    out["ri"]["observed"] = _f(r.dw24_future)
    return out


def _persistence(r):
    leads = []
    for h, k in zip(LEADS_H, [20, 40, 80, 160, 240]):
        u = r.u_motion if np.isfinite(r.u_motion) else 0.0
        v = r.v_motion if np.isfinite(r.v_motion) else 0.0
        lat = r.lat + v * h / KM_PER_DEG
        lon = r.lon + u * h / (KM_PER_DEG * np.cos(np.radians(r.lat)))
        w = float(r.wind) if np.isfinite(r.wind) else 25.0
        leads.append({"lead_h": h, "lat": float(lat), "lon": float(lon), "cone_radius_km": float(k * 1.8), "wind_kt": w,
                      "wind_sd_kt": 10.0, "wind_imd_kt": w * WIND_1MIN_TO_IMD,
                      "imd_class": IMD_CODES[int(imd_class_index(w * WIND_1MIN_TO_IMD))]})
    return leads


def ri_timeline(sid):
    s = storm_rows(sid)
    rm, gm = ri_model(), genesis_model()
    p = rm.prob(s)
    g = gm.prob(s)
    return {"sid": sid, "threshold": rm.threshold, "climatology": rm.base_rate,
            "points": [{"time": t.isoformat(), "ri_probability": float(pi), "observed_dw24": _f(o), "wind_kt": _f(w),
                        "genesis_probability": float(gi) if wi < CS_KT else None}
                       for t, pi, o, w, gi, wi in zip(s.time, p, s.dw24_future, s.wind, g, s.wind_imd_eff.fillna(99))]}
