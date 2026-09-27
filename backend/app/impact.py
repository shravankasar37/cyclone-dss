"""Impact-based outputs: landfall probability / point / time window, population exposure,
fishermen sea-area warnings and SMS previews. All illustrative, derived from the model forecast."""
import os
from datetime import timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from ml.ibtracs import LEADS_H, haversine_km

from . import ROOT
from .alerts import DISTRICTS, gale_radius_km
from .inference import track_ensemble

HOURS = np.arange(0, 73, 1)


@lru_cache
def geo():
    z = np.load(os.path.join(ROOT, "data", "processed", "geo_grids.npz"))
    return {k: z[k] for k in z.files}


def _on_land(lat, lon):
    g = geo()
    iy = np.clip(((np.asarray(lat) - g["lat"][0]) / g["res"] + 0.5).astype(int), 0, len(g["lat"]) - 1)
    ix = np.clip(((np.asarray(lon) - g["lon"][0]) / g["res"] + 0.5).astype(int), 0, len(g["lon"]) - 1)
    return g["land"][iy, ix]


def _hourly(lat0, lon0, w0, lats, lons, winds):
    """Interpolate lead-time points (incl. t=0) to hourly paths. lats etc: (n, leads)."""
    hs = np.array([0] + LEADS_H, float)
    la = np.column_stack([np.full(len(lats), lat0), lats])
    lo = np.column_stack([np.full(len(lats), lon0), lons])
    wi = np.column_stack([np.full(len(lats), w0), winds])
    f = lambda a: np.stack([np.interp(HOURS, hs, r) for r in a])  # noqa: E731
    return f(la), f(lo), f(wi)


def _country_at(lat, lon):
    """Country of the nearest populated land cell within ~30 km (coastline cells can be empty)."""
    g = geo()
    iy = int(np.clip((lat - g["lat"][0]) / g["res"], 0, len(g["lat"]) - 1))
    ix = int(np.clip((lon - g["lon"][0]) / g["res"], 0, len(g["lon"]) - 1))
    win = g["country"][max(iy - 6, 0):iy + 7, max(ix - 6, 0):ix + 7]
    ids, counts = np.unique(win[win > 0], return_counts=True)
    if len(ids):
        return str(g["countries"][ids[np.argmax(counts)] - 1])
    # outside the population grids: name the coast by region
    if lon < 62 and lat > 12:
        return "Arabian Peninsula (Oman/Yemen)"
    if lon < 55:
        return "Horn of Africa"
    return "Island / other coast"


def _nearest_district(lat, lon):
    d = [haversine_km(lat, lon, x["lat"], x["lon"]) for x in DISTRICTS]
    i = int(np.argmin(d))
    return {**{k: DISTRICTS[i][k] for k in ("name", "state")}, "distance_km": float(d[i])}


def landfall(a, row):
    c = a["current"]
    if _on_land(c["lat"], c["lon"]):
        return {"status": "over_land", "message": "Centre is already over land."}
    lat, lon, wind = track_ensemble(row)
    hl, ho, hw = _hourly(c["lat"], c["lon"], c["wind_imd_kt"], lat, lon, wind)
    land = _on_land(hl, ho)
    hit = land.any(1)
    first = np.where(hit, land.argmax(1), -1)
    p = float(hit.mean())
    out = {"status": "possible" if p > 0 else "none", "probability_72h": p, "members": int(len(hit))}
    if p == 0:
        return out
    idx = np.where(hit)[0]
    h = first[idx]
    pts_lat, pts_lon = hl[idx, h], ho[idx, h]
    w = hw[idx, h]
    med_lat, med_lon = float(np.median(pts_lat)), float(np.median(pts_lon))
    t0 = pd.Timestamp(a["time"])
    q10, q50, q90 = np.percentile(h, [10, 50, 90])
    spread = haversine_km(pts_lat, pts_lon, med_lat, med_lon)
    out.update({
        "point": {"lat": med_lat, "lon": med_lon}, "country": _country_at(med_lat, med_lon),
        "nearest_district": _nearest_district(med_lat, med_lon),
        "india_probability": float(np.mean([_country_at(y, x) == "India" for y, x in zip(pts_lat, pts_lon)])) * p,
        "time_median": (t0 + timedelta(hours=float(q50))).isoformat(),
        "window": [(t0 + timedelta(hours=float(q10))).isoformat(), (t0 + timedelta(hours=float(q90))).isoformat()],
        "hours": [float(q10), float(q50), float(q90)],
        "wind_imd_kt": [float(np.percentile(w, 10)), float(np.median(w)), float(np.percentile(w, 90))],
        "spread_km_80pct": float(np.percentile(spread, 80)),
        "member_points": [[float(x), float(y)] for x, y in zip(pts_lon, pts_lat)],
    })
    return out


def exposure(a):
    """People within the damaging-wind envelope of the mean forecast (0-72 h) and within 50 km of landfall."""
    g = geo()
    c, leads = a["current"], a["forecast"]["leads"]
    hl, ho, hw = _hourly(c["lat"], c["lon"], c["wind_imd_kt"], np.array([[x["lat"] for x in leads]]),
                         np.array([[x["lon"] for x in leads]]), np.array([[x["wind_imd_kt"] for x in leads]]))
    hl, ho, hw = hl[0], ho[0], hw[0]
    lat_ax, lon_ax, res = g["lat"], g["lon"], float(g["res"])
    mask = np.zeros(g["pop"].shape, bool)
    first_h = np.full(g["pop"].shape, 999, np.int16)
    for h in range(0, 73, 3):
        if hw[h] < 34:
            continue
        r = gale_radius_km(hw[h])
        dlat = r / 111.2
        dlon = r / (111.2 * max(np.cos(np.radians(hl[h])), 0.2))
        i0, i1 = np.searchsorted(lat_ax, hl[h] - dlat), np.searchsorted(lat_ax, hl[h] + dlat)
        j0, j1 = np.searchsorted(lon_ax, ho[h] - dlon), np.searchsorted(lon_ax, ho[h] + dlon)
        if i0 >= i1 or j0 >= j1:
            continue
        LA, LO = np.meshgrid(lat_ax[i0:i1], lon_ax[j0:j1], indexing="ij")
        inside = haversine_km(LA, LO, hl[h], ho[h]) <= r
        sub = first_h[i0:i1, j0:j1]
        sub[inside & (sub > h)] = h
        mask[i0:i1, j0:j1] |= inside
    pop = g["pop"]
    by_country = {}
    for k, name in enumerate(g["countries"], 1):
        n = float(pop[mask & (g["country"] == k)].sum())
        if n >= 1000:
            by_country[str(name)] = n
    windows = {f"0-{hh}h": float(pop[mask & (first_h <= hh)].sum()) for hh in (24, 48, 72)}
    return {"people_in_gale_envelope": float(pop[mask].sum()), "by_country": by_country, "by_window": windows,
            "method": "WorldPop 2020 population inside illustrative gale radius (>=34 kt) of the mean forecast track"}


# ---------- fishermen sea areas (approximate boxes; illustrative) ----------
SEA_AREAS = [
    ("North Bay of Bengal", 18, 23, 85, 94), ("Westcentral Bay of Bengal", 13, 18, 80, 87), ("Eastcentral Bay of Bengal", 13, 18, 87, 94),
    ("Southwest Bay of Bengal", 8, 13, 80, 86), ("Southeast Bay of Bengal", 5, 13, 86, 93), ("Andaman Sea", 5, 17, 93, 99),
    ("Gulf of Mannar", 7.5, 9.5, 78, 79.8), ("Comorin area", 5, 8, 75, 79),
    ("Northeast Arabian Sea", 18, 24, 64, 73), ("Northwest Arabian Sea", 18, 25, 57, 64), ("Eastcentral Arabian Sea", 12, 18, 67, 74),
    ("Westcentral Arabian Sea", 12, 18, 57, 67), ("Southeast Arabian Sea", 5, 12, 70, 77), ("Southwest Arabian Sea", 5, 12, 51, 70),
    ("Lakshadweep area", 8, 13, 71, 74.5),
]


def _sea_state(w):
    if w >= 64:
        return "Phenomenal to very high"
    if w >= 48:
        return "High to very high"
    if w >= 34:
        return "Rough to very rough"
    return "Rough"


def sea_areas(a):
    c, leads = a["current"], a["forecast"]["leads"]
    hl, ho, hw = _hourly(c["lat"], c["lon"], c["wind_imd_kt"], np.array([[x["lat"] for x in leads]]),
                         np.array([[x["lon"] for x in leads]]), np.array([[x["wind_imd_kt"] for x in leads]]))
    rad = np.interp(HOURS, [0] + LEADS_H, [0] + [x["cone_radius_km"] for x in leads])
    out = []
    for name, la0, la1, lo0, lo1 in SEA_AREAS:
        hits = []
        for h in range(0, 73, 3):
            reach = rad[h] + gale_radius_km(hw[0][h])
            cy, cx = np.clip(hl[0][h], la0, la1), np.clip(ho[0][h], lo0, lo1)  # nearest point of the box
            if haversine_km(hl[0][h], ho[0][h], cy, cx) <= reach and hw[0][h] >= 28:
                hits.append((h, hw[0][h]))
        if not hits:
            continue
        h0, wmax = hits[0][0], max(w for _, w in hits)
        if h0 <= 24:
            action, color = "Fishermen must not venture; those at sea should return to the coast immediately", "Red"
        elif h0 <= 48:
            action, color = "Fishermen advised not to venture from the next 24 h", "Orange"
        else:
            action, color = "Fishermen advised to watch for further warnings", "Yellow"
        out.append({"area": name, "from_h": int(h0), "max_wind_kt": float(wmax), "sea_state": _sea_state(wmax),
                    "action": action, "color": color, "box": [lo0, la0, lo1, la1]})
    return sorted(out, key=lambda x: x["from_h"])


# ---------- SMS / cell-broadcast previews ----------
def _segments(text):
    gsm = all(ord(ch) < 128 for ch in text)
    single, multi = (160, 153) if gsm else (70, 67)
    n = len(text)
    return {"encoding": "GSM-7" if gsm else "UCS-2", "chars": n, "segments": 1 if n <= single else -(-n // multi)}


def sms(a, lf, impacts):
    name = a["name"].title()
    cls = a["current"]["imd_class"]["code"]
    top = impacts[0]["color"] if impacts else None
    where = ", ".join(d["name"] for d in impacts[:3]) if impacts else ""
    when = f" ~{int(lf['hours'][1])}h" if lf.get("hours") else ""
    en = (f"CYCLONE {name} ({cls}) {top.upper()} ALERT: {where}. Landfall{when}. Move to shelter if told. "
          f"Fishermen stay ashore. -Prototype, not official" if top else f"CYCLONE {name} ({cls}): no coastal alert now. -Prototype")
    hi = (f"चक्रवात {name} ({cls}) {top} अलर्ट: {where}। निर्देश पर आश्रय जाएँ। मछुआरे समुद्र में न जाएँ।" if top else
          f"चक्रवात {name} ({cls}): अभी कोई तटीय अलर्ट नहीं।")
    gu = (f"વાવાઝોડું {name} ({cls}) {top} એલર્ટ: {where}. સૂચના મળે તો આશ્રયસ્થાને જાઓ. માછીમારો દરિયામાં ન જાય." if top else
          f"વાવાઝોડું {name} ({cls}): હાલ કોઈ દરિયાકાંઠા એલર્ટ નથી.")
    return [{"lang": l, "text": t, **_segments(t)} for l, t in [("en", en), ("hi", hi), ("gu", gu)]]
