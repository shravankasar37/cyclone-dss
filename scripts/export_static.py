"""Export the whole prototype as static files for free, always-on hosting (Vercel, no backend).

Every value is produced by the real models through the real API (FastAPI TestClient); it is only
computed ahead of time. Output: frontend/public/data/...
Run (from the project root):  .venv\\Scripts\\python scripts\\export_static.py
"""
import base64
import io
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "frontend", "public", "data")
MIN_SEASON = int(os.environ.get("MIN_SEASON", "2010"))
WORKERS = int(os.environ.get("WORKERS", "3"))
sys.path.insert(0, os.path.join(ROOT, "backend"))
os.environ["GEMINI_API_KEY"] = ""  # deterministic template bulletins; the key never leaves the laptop

_client = None


def client():
    global _client
    if _client is None:
        from fastapi.testclient import TestClient
        from app.main import app
        _client = TestClient(app)
    return _client


def get(path):
    r = client().get(path)
    r.raise_for_status()
    return r


def dump(rel, obj):
    p = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)


def save_image(data_url, rel, quality=82):
    """data:image/png;base64,... -> WebP file (keeps transparency); returns the public URL."""
    from PIL import Image
    raw = base64.b64decode(data_url.split(",", 1)[1])
    im = Image.open(io.BytesIO(raw))
    p = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    im.save(p, "WEBP", quality=quality, method=6)
    return "/data/" + rel.replace(os.sep, "/")


def export_storm(sid):
    t0 = time.time()
    track = get(f"/api/storms/{sid}/track").json()
    bundle = {"track": track, "ri_timeline": get(f"/api/storms/{sid}/ri-timeline").json(), "times": {}}
    errors = []
    for p in track["points"]:
        t = p["time"]
        try:
            bundle["times"][t] = {
                "analysis": get(f"/api/storms/{sid}/analysis?time={t}").json(),
                "impact": get(f"/api/storms/{sid}/impact?time={t}").json(),
                "cap": get(f"/api/storms/{sid}/cap.xml?time={t}").text,
                "bulletin": get(f"/api/storms/{sid}/bulletin?time={t}").json(),
            }
        except Exception as e:  # recorded and reported, never silently dropped
            errors.append(f"{sid} {t}: {type(e).__name__}: {str(e)[:120]}")
    dump(f"storms/{sid}.json", bundle)
    return sid, track["name"], len(track["points"]), time.time() - t0, errors


def export_satellite():
    storms = get("/api/satellite/storms").json()
    dump("sat/storms.json", storms)
    n = 0
    for s in storms:
        frames = get(f"/api/satellite/frames?tcir_sid={s['tcir_sid']}").json()
        series = get(f"/api/satellite/series?tcir_sid={s['tcir_sid']}").json()
        dump(f"sat/{s['tcir_sid']}.json", {"frames": frames, "series": series})
        for f in frames:
            k = f["k"]
            d = get(f"/api/satellite/frames/{k}").json()
            d["image_png"] = save_image(d["image_png"], os.path.join("sat", "img", f"{k}_ir.webp"))
            d["gradcam_png"] = save_image(d["gradcam_png"], os.path.join("sat", "img", f"{k}_cam.webp"))
            dump(f"sat/frame/{k}.json", d)
            n += 1
    return n


def export_basin():
    st = get("/api/basin/storms").json()
    dump("basin/storms.json", st)
    if not st.get("available"):
        return 0
    n = 0
    for s in st["storms"]:
        d = get(f"/api/basin/storms/{s['storm']}").json()
        dump(f"basin/{s['storm']}.json", d)
        for k in d["ks"]:
            sc = get(f"/api/basin/scene/{k}").json()
            sc["png"] = save_image(sc["png"], os.path.join("basin", "scene", f"{k}.webp"), quality=80)
            dump(f"basin/scene/{k}.json", sc)
            n += 1
    return n


def main():
    t0 = time.time()
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    storms = [s for s in get("/api/storms").json() if s["season"] >= MIN_SEASON]
    dump("storms.json", storms)
    dump("metrics.json", get("/api/metrics").json())
    with open(os.path.join(OUT, "intensity_eval.png"), "wb") as f:
        f.write(get("/api/metrics/intensity_eval.png").content)
    print(f"{len(storms)} storms since {MIN_SEASON}; exporting with {WORKERS} workers ...", flush=True)
    with Pool(WORKERS) as pool:
        all_errors = []
        for i, (sid, name, n, dt, errs) in enumerate(pool.imap_unordered(export_storm, [s["sid"] for s in storms]), 1):
            all_errors += errs
            print(f"  [{i}/{len(storms)}] {name} {sid}: {n} times, {dt:.0f}s{f', {len(errs)} ERRORS' if errs else ''}", flush=True)
    for e in all_errors:
        print("  ERROR", e)
    print("satellite frames:", export_satellite(), flush=True)
    print("basin scenes:", export_basin(), flush=True)
    dump("manifest.json", {"generated": time.strftime("%Y-%m-%d %H:%M"), "min_season": MIN_SEASON, "storms": len(storms),
                           "errors": all_errors,
                           "note": "Pre-computed by the prototype's models through its API; static replay edition."})
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs)
    files = sum(len(fs) for _, _, fs in os.walk(OUT))
    print(f"done in {(time.time() - t0) / 60:.1f} min: {files} files, {size / 1e6:.0f} MB -> {OUT}")


if __name__ == "__main__":
    main()
