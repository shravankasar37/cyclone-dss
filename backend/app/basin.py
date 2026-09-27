"""Full-basin identification: detect every system in each GridSat scene, link detections into
tracks, estimate intensity (IR+WV multi-channel CNN) and 24 h RI risk (ConvLSTM) from images alone,
and score against IBTrACS best track. Results are computed once per demo storm and cached."""
import base64
import io
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from ml.cyclone_scale import IMD_CODES, WIND_1MIN_TO_IMD, imd_class_index, round_t, t_to_wind
from ml.gridsat import IR_RANGE, crop, u8_to_bt
from ml.ibtracs import haversine_km
from ml.nets import DET_CLASSES, BasinDetector, MultiChannelIntensityNet, SeqIntensityNet, heatmap_peaks

from . import ROOT
from .config import WEIGHTS
from .inference import storm_table

ADV = os.environ.get("ADV_WEIGHTS_DIR", WEIGHTS)
CACHE_DIR = os.path.join(ROOT, "data", "processed", "basin_cache")
CACHE_VERSION = "v4"
CONFIRM_FIXES = 6  # a track is "confirmed" after persisting 15 h (6 three-hourly fixes)  # bump when the tracking / scoring logic changes
REQUIRED = ["demo_scenes.npz", "detector.pt", "mc_intensity_model.pt", "seq_model.pt"]


def available():
    return all(os.path.exists(os.path.join(ADV, f)) for f in REQUIRED)


def _load(name):
    return torch.load(os.path.join(ADV, name), map_location="cpu", weights_only=False)


@lru_cache
def scenes():
    z = np.load(os.path.join(ADV, "demo_scenes.npz"))
    meta = pd.DataFrame({"k": np.arange(len(z["time"])), "time": pd.to_datetime(z["time"]), "storm": z["storm"]})
    return z["ir"], z["wv"], z["lat"], z["lon"], meta


@lru_cache
def detector():
    ck = _load("detector.pt")
    m = BasinDetector()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    m.cfg = ck["config"]
    return m


@lru_cache
def mc_model():
    ck = _load("mc_intensity_model.pt")
    m = MultiChannelIntensityNet(n_cls=len(ck["config"]["classes"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    m.cfg = ck["config"]
    return m


@lru_cache
def seq_model():
    ck = _load("seq_model.pt")
    m = SeqIntensityNet()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    m.cfg = ck["config"]
    return m


# ---------- per-scene inference ----------
EXTEND_THR = 0.12  # weaker peaks may continue an existing track, but only confident ones start a new track


@lru_cache
def wv_missing():
    """Per-scene flag: GridSat has no water-vapour data (e.g. all of 2024)."""
    _, wv, _, _, _ = scenes()
    return (wv.reshape(len(wv), -1) == 0).mean(1) > 0.9


@lru_cache
def ir_to_wv_lut():
    """IR -> WV brightness lookup learned on scenes that have both (corr 0.87); used only when WV is missing."""
    p = os.path.join(ROOT, "data", "processed", "ir_to_wv_lut.npy")
    return np.load(p) if os.path.exists(p) else None


def detect(k):
    ir, wv, lat, lon, _ = scenes()
    m = detector()
    hp, wp = m.cfg["pad"]
    x = np.zeros((2, hp, wp), np.float32)
    w = wv[k]
    if wv_missing()[k] and ir_to_wv_lut() is not None:
        w = ir_to_wv_lut()[ir[k]]
    x[0, :ir.shape[1], :ir.shape[2]], x[1, :wv.shape[1], :wv.shape[2]] = ir[k], w
    with torch.no_grad():
        logits = m((torch.from_numpy(x)[None] / 255 - 0.5) / 0.25)[0]
    s, res = m.cfg["stride"], m.cfg["res"]
    dets = sorted([(sc, c, float(lat[0] + (y + 0.5) * s * res), float(lon[0] + (xx + 0.5) * s * res))
                   for c, y, xx, sc in heatmap_peaks(logits, EXTEND_THR)], reverse=True)
    merged = []
    for d in dets:
        if all(haversine_km(d[2], d[3], o[2], o[3]) > 150 for o in merged):
            merged.append(d)
    thr = m.cfg["threshold"]
    return [{"score": sc, "cls": DET_CLASSES[c], "lat": la, "lon": lo, "strong": sc >= thr} for sc, c, la, lo in merged]


def _crop_pair(k, la, lo):
    ir, wv, lat, lon, _ = scenes()
    i, j = int(np.abs(lat - la).argmin()), int(np.abs(lon - lo).argmin())
    return crop(ir[k], i, j), crop(wv[k], i, j)


def intensity(k, la, lo):
    m = mc_model()
    cfg = m.cfg
    ci, cw = _crop_pair(k, la, lo)
    t = torch.from_numpy(np.stack([ci, cw]).astype(np.float32))[None]
    t = F.interpolate(t, size=(cfg["img_size"],) * 2, mode="area") / 255.0
    ones = torch.ones_like(t[:, :1])
    wm = 0.0 if wv_missing()[k] else 1.0  # IR-only mode when water vapour is missing (the model was trained for this)
    x = torch.cat([(t[:, :1] - 0.5) / 0.25, (t[:, 1:2] - 0.5) / 0.25 * wm, torch.zeros_like(ones), ones * wm, torch.zeros_like(ones)], 1)
    with torch.no_grad():
        outs = [m(torch.rot90(x, r, (2, 3))) for r in range(4)]
    v = float(torch.stack([o[0] for o in outs]).mean()) * cfg["v_sd"] + cfg["v_mu"]
    tn = float(torch.stack([o[1] for o in outs]).mean()) * cfg["t_sd"] + cfg["t_mu"]
    probs = torch.stack([o[2].softmax(1) for o in outs]).mean(0)[0].numpy()
    w_imd = v * WIND_1MIN_TO_IMD
    return {"wind_kt": v, "wind_imd_kt": w_imd, "t_number": float(round_t(np.clip(tn, 1, 8))),
            "dvorak_wind_kt": float(t_to_wind(np.clip(tn, 1, 8))), "imd_class": IMD_CODES[int(imd_class_index(w_imd))],
            "class_probs": {c: float(p) for c, p in zip(IMD_CODES, probs)}}


def seq_outlook(frames_k, centers, winds):
    """frames_k: scene indices oldest->now (8), centers: [(lat, lon)], winds: CNN winds (kt) along the track."""
    m = seq_model()
    img = m.cfg["img"]
    fr = []
    for k, (la, lo) in zip(frames_k, centers):
        ci, cw = _crop_pair(k, la, lo)
        t = torch.from_numpy(np.stack([ci, cw]).astype(np.float32))[None]
        fr.append(F.interpolate(t, size=(img, img), mode="area")[0] / 255.0)
    frames = ((torch.stack(fr) - 0.5) / 0.25)[None]
    scal = torch.tensor([[winds[-1] / 100.0, (winds[-1] - winds[0]) / 30.0]], dtype=torch.float32)
    with torch.no_grad():
        d, lg = m(frames, scal)
    return {"dv24_kt": float(d) * 20, "ri_probability": float(torch.sigmoid(lg))}


# ---------- whole-storm processing: detection -> tracking -> intensity -> ConvLSTM ----------
MAX_GAP_H = 12


def _gate_km(gap_h):
    """Max centre displacement allowed between fixes: ~40 km/h storm motion plus detector position noise."""
    return 40 * gap_h + 90


def _link(det_by_scene, times):
    """Greedy nearest-neighbour tracker with hysteresis: strong detections start tracks, weak ones only extend."""
    tracks = []
    for k, t in zip(det_by_scene, times):
        claimed = set()
        for d in sorted(det_by_scene[k], key=lambda d: -d["score"]):
            best = None
            for tr in tracks:
                last = tr["points"][-1]
                gap = (t - pd.Timestamp(last["time"])).total_seconds() / 3600
                dist = haversine_km(last["lat"], last["lon"], d["lat"], d["lon"])
                if tr["id"] not in claimed and 0 < gap <= MAX_GAP_H and dist <= _gate_km(gap) and (best is None or dist < best[0]):
                    best = (dist, tr)
            p = {**d, "k": int(k), "time": t.isoformat()}
            if best:
                best[1]["points"].append(p)
                claimed.add(best[1]["id"])
            elif d["strong"]:
                tracks.append({"id": len(tracks), "points": [p]})
                claimed.add(tracks[-1]["id"])
    # merge fragments: a track that starts shortly after another ends, near where it ended
    tracks.sort(key=lambda tr: tr["points"][0]["time"])
    merged = []
    for tr in tracks:
        head = tr["points"][0]
        for m in merged:
            tail = m["points"][-1]
            gap = (pd.Timestamp(head["time"]) - pd.Timestamp(tail["time"])).total_seconds() / 3600
            if 0 < gap <= 2 * MAX_GAP_H and haversine_km(tail["lat"], tail["lon"], head["lat"], head["lon"]) <= _gate_km(gap):
                m["points"].extend(tr["points"])
                break
        else:
            merged.append(tr)
    keep = [t for t in merged if len(t["points"]) >= 3 and any(p["strong"] for p in t["points"])]
    for i, t in enumerate(keep):
        t["id"] = i
    return keep


def _truth(times):
    ib = storm_table()[["sid", "name", "time", "lat", "lon", "wind", "wind_imd_eff"]]
    return ib[ib.time.isin(times)]


def _weights_signature():
    """Identifies the model files by size - survives git checkouts, which reset file dates."""
    return {f: os.path.getsize(os.path.join(ADV, f)) for f in REQUIRED}


def process_storm(storm):
    path = os.path.join(CACHE_DIR, f"{storm}.{CACHE_VERSION}.json")
    if os.path.exists(path):
        cached = json.load(open(path))
        if cached.get("weights") == _weights_signature():
            return cached
    _, _, _, _, meta = scenes()
    s = meta[meta.storm == storm].sort_values("time")
    times = list(s.time)
    dets = {int(k): detect(int(k)) for k in s.k}
    tracks = _link(dets, times)
    for tr in tracks:
        for p in tr["points"]:
            p.update(intensity(p["k"], p["lat"], p["lon"]))
        for n, p in enumerate(tr["points"]):
            hist = tr["points"][max(0, n - 7):n + 1]
            ks = [h["k"] for h in hist]
            # need 8 consecutive 3-hourly frames of this track
            if len(hist) == 8 and all(pd.Timestamp(b["time"]) - pd.Timestamp(a["time"]) == pd.Timedelta(hours=3)
                                      for a, b in zip(hist[:-1], hist[1:])):
                p["seq"] = seq_outlook(ks, [(h["lat"], h["lon"]) for h in hist], [h["wind_kt"] for h in hist])

    # score against best track: which track is this storm, position error, detection lead time
    truth = _truth(times)
    named = storm_table()
    target = named[(named.name == storm) & named.time.isin(times)]
    sid = target.sid.mode().iloc[0] if len(target) else None
    tt = truth[truth.sid == sid] if sid else truth.iloc[:0]
    main = None
    best_hits = 0
    for tr in tracks:
        errs = []
        for p in tr["points"]:
            r = tt[tt.time == pd.Timestamp(p["time"])]
            if len(r):
                e = float(haversine_km(p["lat"], p["lon"], r.lat.iloc[0], r.lon.iloc[0]))
                p["truth"] = {"lat": float(r.lat.iloc[0]), "lon": float(r.lon.iloc[0]), "wind_imd_kt": float(r.wind_imd_eff.iloc[0]),
                              "wind_kt": float(r.wind.iloc[0]) if np.isfinite(r.wind.iloc[0]) else None, "error_km": e}
                errs.append(e)
        hits = sum(e <= 200 for e in errs)
        tr["matched_points"], tr["median_error_km"] = hits, float(np.median(errs)) if errs else None
        if hits > best_hits:
            best_hits, main = hits, tr
    # other tracks: do they follow some other real system in IBTrACS, or are they false alarms?
    for tr in tracks:
        tr["confirmed"] = len(tr["points"]) >= CONFIRM_FIXES
        tr["matches_any_system"] = any(
            (haversine_km(p["lat"], p["lon"], truth[truth.time == pd.Timestamp(p["time"])].lat.values,
                          truth[truth.time == pd.Timestamp(p["time"])].lon.values) <= 200).any()
            for p in tr["points"] if (truth.time == pd.Timestamp(p["time"])).any())
    summary = {"storm": storm, "scenes": len(times), "start": times[0].isoformat(), "end": times[-1].isoformat(),
               "tracks": len(tracks), "main_track": main["id"] if main else None,
               "tracks_real": int(sum(t["matches_any_system"] for t in tracks)),
               "tracks_false": int(sum(not t["matches_any_system"] for t in tracks)),
               "confirmed_real": int(sum(t["confirmed"] and t["matches_any_system"] for t in tracks)),
               "confirmed_false": int(sum(t["confirmed"] and not t["matches_any_system"] for t in tracks)),
               "wv_missing": bool(wv_missing()[s.k.values].all())}
    # storm-level scores use every detection of this storm, on whichever track fragment it landed
    hits = [p for tr in tracks for p in tr["points"] if p.get("truth", {}).get("error_km", 1e9) <= 200]
    if hits and len(tt):
        first_det = min(pd.Timestamp(p["time"]) for p in hits)
        first_d = tt[tt.wind_imd_eff >= 17].time.min()
        first_cs = tt[tt.wind_imd_eff >= 34].time.min()
        synoptic = tt[tt.time.dt.hour % 6 == 0]
        hit_times = {pd.Timestamp(p["time"]) for p in hits}
        found = sum(t in hit_times for t in synoptic.time)
        winds = [(p["wind_kt"], p["truth"]["wind_kt"]) for p in hits if p["truth"].get("wind_kt") is not None]
        summary.update({
            "first_detection": first_det.isoformat(),
            "hours_before_depression": (first_d - first_det).total_seconds() / 3600 if pd.notna(first_d) else None,
            "hours_before_cyclonic_storm": (first_cs - first_det).total_seconds() / 3600 if pd.notna(first_cs) else None,
            "detection_rate": found / max(len(synoptic), 1),
            "median_position_error_km": float(np.median([p["truth"]["error_km"] for p in hits])),
            "intensity_mae_kt": float(np.mean([abs(a - b) for a, b in winds])) if winds else None,
            "best_track": [{"time": t.isoformat(), "lat": float(la), "lon": float(lo), "wind_imd_kt": float(w)}
                           for t, la, lo, w in zip(tt.time, tt.lat, tt.lon, tt.wind_imd_eff)],
        })
    out = {"summary": summary, "tracks": tracks, "detections": {str(k): v for k, v in dets.items()},
           "times": [t.isoformat() for t in times], "ks": [int(k) for k in s.k]}
    out["weights"] = _weights_signature()
    try:  # a read-only deployment still works, it just recomputes after restarts
        os.makedirs(CACHE_DIR, exist_ok=True)
        json.dump(out, open(path, "w"), default=float)
    except OSError:
        pass
    return out


# ---------- rendering ----------
def _ir_palette():
    """Enhanced IR: transparent clear sky, grey mid-level cloud, colours for deep convection."""
    bt = np.linspace(IR_RANGE[1], IR_RANGE[0], 256)
    rgba = np.zeros((256, 4), np.uint8)
    for i, b in enumerate(bt):
        if b > 285:
            rgba[i] = (0, 0, 0, 0)
        elif b > 240:
            g = int(90 + (285 - b) / 45 * 150)
            rgba[i] = (g, g, g, int(min(255, (285 - b) / 20 * 200)))
        else:
            stops = [(240, (120, 190, 255)), (225, (40, 110, 240)), (210, (40, 200, 90)), (200, (250, 220, 40)),
                     (190, (240, 90, 30)), (180, (200, 20, 40))]
            for (b0, c0), (b1, c1) in zip(stops[:-1], stops[1:]):
                if b1 <= b <= b0:
                    f = (b0 - b) / (b0 - b1)
                    rgba[i] = (*[int(a + (c - a) * f) for a, c in zip(c0, c1)], 235)
                    break
            else:
                rgba[i] = (200, 20, 40, 235)
    return rgba


PALETTE = _ir_palette()


@lru_cache(maxsize=256)
def scene_png(k):
    ir, _, lat, lon, _ = scenes()
    rgba = PALETTE[ir[k][::-1]]  # north-up
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    res = float(lat[1] - lat[0])
    bounds = [float(lon[0] - res / 2), float(lat[0] - res / 2), float(lon[-1] + res / 2), float(lat[-1] + res / 2)]
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), bounds


def bt_at(k, la, lo):
    ir, _, lat, lon, _ = scenes()
    return float(u8_to_bt(ir[k][int(np.abs(lat - la).argmin()), int(np.abs(lon - lo).argmin())], IR_RANGE))
