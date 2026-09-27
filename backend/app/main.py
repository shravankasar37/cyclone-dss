"""PS 26070 cyclone decision-support API.  Run: uvicorn app.main:app --reload (from backend/)"""
import json
import os

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from . import db
from .alerts import build_cap, district_impacts
from .analysis import analysis, ri_timeline, select_row, storms_summary, track_payload
from .impact import exposure, landfall, sea_areas, sms
from .bulletin import generate as generate_bulletin
from .config import CORS_ORIGINS, GEMINI_API_KEY, WEIGHTS
from .frames import frames, storm_list
from .inference import analyze_image, gray_png, intensity_model, load_uploaded_image

app = FastAPI(title="PS 26070 Cyclone Decision-Support API",
              description="Research prototype for SIH 2026 PS 26070. Not an official IMD/NDMA product.",
              version="0.1.0")
# local dev + LAN access (e.g. demo laptop on a hotspot); production origins come from CORS_ORIGINS
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_origin_regex=r"http://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+):3000",
                   allow_methods=["*"], allow_headers=["*"])


def _clean(o):
    """Make numpy/NaN values JSON-safe."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


@app.get("/api/health")
def health():
    return {"status": "ok", "models": sorted(f for f in os.listdir(WEIGHTS) if f.endswith((".pt", ".txt"))),
            "llm_bulletins": bool(GEMINI_API_KEY)}


@app.get("/api/storms")
def storms(min_season: int = 1990):
    g = storms_summary(min_season)
    g = g.assign(start=g.start.astype(str), end=g.end.astype(str))
    return _clean(g.to_dict("records"))


@app.get("/api/storms/{sid}/track")
def storm_track(sid: str):
    try:
        return _clean(track_payload(sid))
    except KeyError:
        raise HTTPException(404, "unknown storm")


def _analysis(sid, time):
    try:
        return analysis(sid, time)
    except KeyError:
        raise HTTPException(404, "unknown storm")


@app.get("/api/storms/{sid}/analysis")
def storm_analysis(sid: str, time: str | None = None):
    a = _analysis(sid, time)
    a["impacts"] = district_impacts(a)
    db.log("analysis", f"{a['name']} ({sid})", a["time"],
           {"wind_imd_kt": round(a["current"]["wind_imd_kt"]), "ri_p": round(a["ri"]["probability"], 3),
            "districts": len(a["impacts"]), "class": a["current"]["imd_class"]["code"]})
    return _clean(a)


@app.get("/api/storms/{sid}/impact")
def storm_impact(sid: str, time: str | None = None):
    """Landfall window, population exposure, fishermen sea-area warnings and SMS previews."""
    a = _analysis(sid, time)
    row, forecastable = select_row(sid, a["time"])
    lf = landfall(a, row) if forecastable else {"status": "insufficient_history",
                                                 "message": "Needs 18 h of track history for the ensemble."}
    impacts = district_impacts(a)
    return _clean({"time": a["time"], "landfall": lf, "exposure": exposure(a), "sea_areas": sea_areas(a),
                   "sms": sms(a, lf, impacts)})


@app.get("/api/storms/{sid}/ri-timeline")
def storm_ri_timeline(sid: str):
    try:
        return _clean(ri_timeline(sid))
    except KeyError:
        raise HTTPException(404, "unknown storm")


@app.get("/api/storms/{sid}/cap.xml")
def storm_cap(sid: str, time: str | None = None, langs: str = "en,hi,gu"):
    a = _analysis(sid, time)
    xml = build_cap(a, district_impacts(a), tuple(l for l in langs.split(",") if l in ("en", "hi", "gu")))
    db.log("cap", f"{a['name']} ({sid})", a["time"], {"bytes": len(xml)})
    return Response(xml, media_type="application/xml")


@app.get("/api/storms/{sid}/bulletin")
def storm_bulletin(sid: str, time: str | None = None):
    a = _analysis(sid, time)
    return generate_bulletin(_clean(a), district_impacts(a))


@app.get("/api/satellite/storms")
def satellite_storms():
    g = storm_list()
    g = g.assign(start=g.start.astype(str), end=g.end.astype(str))
    return _clean(g.to_dict("records"))


@app.get("/api/satellite/frames")
def satellite_frames(tcir_sid: str):
    _, m = frames()
    s = m[m.tcir_sid == tcir_sid].sort_values("time")
    return _clean([{"k": int(r.k), "time": r.time.isoformat(), "lat": r.lat, "lon": r.lon, "best_track_kt": r.vmax}
                   for r in s.itertuples()])


@app.get("/api/satellite/series")
def satellite_series(tcir_sid: str):
    """CNN wind and Dvorak-derived wind for every frame of a storm vs best track."""
    from ml.cyclone_scale import t_to_wind
    _, m = frames()
    s = m[m.tcir_sid == tcir_sid].sort_values("time")
    return _clean([{"k": int(r.k), "time": r.time.isoformat(), "best_track_kt": r.vmax, "cnn_kt": r.cnn_vmax,
                    "dvorak_kt": float(t_to_wind(r.cnn_tnum))} for r in s.itertuples()])


@app.get("/api/satellite/frames/{k}")
def satellite_frame(k: int):
    imgs, m = frames()
    if not 0 <= k < len(imgs):
        raise HTTPException(404, "unknown frame")
    r = m[m.k == k].iloc[0]
    from ml.adt_features import FEATURE_NAMES
    res = analyze_image(imgs[k], adt={f: r[f] for f in FEATURE_NAMES})
    if "ri_prob" in r and np.isfinite(r.get("ri_prob", np.nan)):
        res["seq"] = {"dv24_kt": r.dv24_pred, "ri_probability": r.ri_prob, "observed_dv24_kt": r.dv24_obs}
    res.update({"k": k, "name": r["name"], "tcir_sid": r.tcir_sid, "time": r.time.isoformat(), "lat": r.lat, "lon": r.lon,
                "best_track_kt": r.vmax, "image_png": gray_png(imgs[k]), "held_out": bool(r.time.year >= 2016)})
    db.log("satellite", r["name"], r.time.isoformat(), {"cnn_kt": round(res["cnn_wind_kt"], 1), "best_track_kt": float(r.vmax),
                                                        "t": res["t_number"]})
    return _clean(res)


@app.post("/api/satellite/upload")
async def satellite_upload(file: UploadFile = File(...), insat_calibration: bool = Query(True)):
    data = await file.read()
    if len(data) > 10_000_000:
        raise HTTPException(413, "image too large")
    try:
        u8 = load_uploaded_image(data, intensity_model().cfg["img_size"])
    except Exception:
        raise HTTPException(400, "could not read image")
    res = analyze_image(u8, calibrate_insat=insat_calibration)
    res["image_png"] = gray_png(u8)
    db.log("upload", file.filename or "upload", None, {"cnn_kt": round(res["cnn_wind_kt"], 1), "t": res["t_number"]})
    return _clean(res)


@app.get("/api/basin/storms")
def basin_storms():
    """Demo storms with full-basin GridSat scenes (needs the advanced Colab outputs)."""
    from . import basin
    if not basin.available():
        return {"available": False, "storms": [], "missing": [f for f in basin.REQUIRED if not os.path.exists(os.path.join(basin.ADV, f))]}
    _, _, _, _, meta = basin.scenes()
    g = meta.groupby("storm").agg(scenes=("k", "size"), start=("time", "min"), end=("time", "max")).reset_index().sort_values("start")
    return {"available": True, "storms": _clean(g.assign(start=g.start.astype(str), end=g.end.astype(str)).to_dict("records"))}


@app.get("/api/basin/storms/{storm}")
def basin_storm(storm: str):
    """Detections, automatic tracks with image-based intensity and ConvLSTM outlook, and best-track scoring."""
    from . import basin
    if not basin.available():
        raise HTTPException(503, "advanced models not installed yet")
    if storm not in set(basin.scenes()[4].storm):
        raise HTTPException(404, "unknown demo storm")
    return _clean(basin.process_storm(storm))


@app.get("/api/basin/scene/{k}")
def basin_scene(k: int):
    from . import basin
    if not basin.available():
        raise HTTPException(503, "advanced models not installed yet")
    meta = basin.scenes()[4]
    if not 0 <= k < len(meta):
        raise HTTPException(404, "unknown scene")
    png, bounds = basin.scene_png(k)
    return {"k": k, "time": meta.time.iloc[k].isoformat(), "storm": meta.storm.iloc[k], "png": png, "bounds": bounds}


@app.get("/api/metrics")
def metrics():
    out = {}
    for name in ["metrics", "ri_metrics", "track_metrics", "genesis_metrics", "metrics_multichannel",
                 "metrics_convlstm", "metrics_detector"]:
        p = os.path.join(WEIGHTS, f"{name}.json")
        if os.path.exists(p):
            out[name] = json.load(open(p))
    return _clean(out)


@app.get("/api/metrics/intensity_eval.png")
def metrics_png():
    return FileResponse(os.path.join(WEIGHTS, "intensity_eval.png"), media_type="image/png")


@app.get("/api/history")
def history(limit: int = 30):
    return db.recent(limit)
