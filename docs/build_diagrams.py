"""Render the PPT/PDF diagrams (1600x900, 16:9) from HTML with headless Edge.
Run: python docs/build_diagrams.py   ->  docs/diagrams/*.png"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "diagrams", "src")
OUT = os.path.join(HERE, "diagrams")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
os.makedirs(SRC, exist_ok=True)

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{width:1600px;height:900px;background:#fff;font-family:'Segoe UI',Arial,sans-serif;color:#1d2433;padding:38px 46px;overflow:hidden}
h1{font-size:40px;font-weight:800;color:#1f3f7a;letter-spacing:-.5px}
.sub{font-size:21px;color:#5b6679;margin-top:6px}
.row{display:flex;gap:16px;align-items:stretch}
.col{flex:1;border-radius:18px;padding:16px 16px 14px;border:2.5px solid;display:flex;flex-direction:column;gap:9px}
.col h3{font-size:21px;font-weight:800;text-transform:uppercase;letter-spacing:.5px;text-align:center;padding:6px 0;border-radius:10px;color:#fff}
.box{background:#fff;border-radius:12px;padding:9px 11px;font-size:17px;line-height:1.3;border:1.5px solid #d9dfe8}
.box b{font-size:17.5px}
.small{font-size:14.5px;color:#5b6679}
.arrow{align-self:center;font-size:40px;color:#8a94a6;font-weight:300}
.c1{border-color:#2f6fd0;background:#f0f5fd}.c1 h3{background:#2f6fd0}
.c2{border-color:#3a9b5c;background:#effaf3}.c2 h3{background:#3a9b5c}
.c3{border-color:#e0a100;background:#fffaeb}.c3 h3{background:#d49400}
.c4{border-color:#e5484d;background:#fff2f2}.c4 h3{background:#e5484d}
.c5{border-color:#3b82c4;background:#eef6fc}.c5 h3{background:#3b82c4}
.c6{border-color:#8a5cd6;background:#f6f1fe}.c6 h3{background:#8a5cd6}
.new{display:inline-block;background:#ff7a1a;color:#fff;font-size:12px;font-weight:800;border-radius:6px;padding:1px 6px;margin-left:4px;vertical-align:2px}
.foot{position:absolute;bottom:22px;left:46px;right:46px;font-size:15px;color:#7a8597}
table{border-collapse:collapse;width:100%}
.arch .col h3{font-size:17px;padding:5px 0}
.arch .box{font-size:15.5px;padding:7px 10px}
.arch .box b{font-size:16px}
.arch .small{font-size:13.5px}
.tall td{padding-top:17px!important;padding-bottom:17px!important}
.tall th{padding:14px 12px}
"""


def page(name, body, extra_css=""):
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}{extra_css}</style></head><body>{body}</body></html>"
    p = os.path.join(SRC, f"{name}.html")
    open(p, "w", encoding="utf-8").write(html)
    out = os.path.join(OUT, f"{name}.png")
    subprocess.run([EDGE, "--headless=new", "--hide-scrollbars", "--window-size=1600,900", "--force-device-scale-factor=1",
                    f"--screenshot={out}", "file:///" + p.replace("\\", "/")], capture_output=True, timeout=120)
    print(name, os.path.getsize(out) if os.path.exists(out) else "FAILED")


# 1. System architecture -------------------------------------------------------------------------
page("01_system_architecture", """
<h1>System Architecture</h1>
<div class="sub">Multi-source satellite + best-track + reanalysis data → six AI models → decision layer → API → single-screen dashboard</div>
<div class="row arch" style="margin-top:22px;height:705px">
 <div class="col c1"><h3>1 · Data sources</h3>
  <div class="box"><b>GridSat-B1</b> (NOAA)<br>basin-wide IR + water vapour, 3-hourly, 0.07°</div>
  <div class="box"><b>TCIR</b> benchmark<br>storm-centred IR · WV · microwave, 2003–16</div>
  <div class="box"><b>INSAT-3D</b> (MOSDAC/Kaggle)<br>external test + live-feed slot</div>
  <div class="box"><b>IBTrACS</b> (NOAA)<br>track, wind, pressure 1982–2025</div>
  <div class="box"><b>ERA5</b> SST, shear, humidity · <b>WorldPop</b> population · <b>Natural Earth</b> coast</div>
 </div>
 <div class="arrow">➜</div>
 <div class="col c2"><h3>2 · Preparation</h3>
  <div class="box">Brightness temp → normalised image, 14° storm crops</div>
  <div class="box">Southern-hemisphere storms mirrored to NIO rotation</div>
  <div class="box">IBTrACS ↔ image matching by time + position</div>
  <div class="box">IMD scale labels (D → SuCS) + Dvorak T-numbers</div>
  <div class="box">ADT-lite cloud features (eye, ring, cold cover)</div>
  <div class="box"><b>Split by storm & year</b> — no leakage</div>
 </div>
 <div class="arrow">➜</div>
 <div class="col c3" style="flex:1.25"><h3>3 · AI models</h3>
  <div class="box"><b>Basin detector</b> ResNet-18 heatmap <span class="new">NEW</span><br><span class="small">finds every system in the raw scene</span></div>
  <div class="box"><b>Intensity CNN</b> EfficientNet-B0<br><span class="small">wind · Dvorak T · IMD class · Grad-CAM</span></div>
  <div class="box"><b>ConvLSTM</b> 8-frame image sequence<br><span class="small">24 h change (experimental RI)</span></div>
  <div class="box"><b>RI early warning</b> LightGBM <span class="new">NEW</span><br><span class="small">P(+30 kt / 24 h), SHAP reasons</span></div>
  <div class="box"><b>Genesis</b> LightGBM <span class="new">NEW</span> · <b>Track</b> LSTM + calibrated cone</div>
 </div>
 <div class="arrow">➜</div>
 <div class="col c4"><h3>4 · Decision layer</h3>
  <div class="box">Landfall probability, place & time window (50-member ensemble)</div>
  <div class="box">Population exposed (gale envelope)</div>
  <div class="box">District staging: Watch / Alert / Warning</div>
  <div class="box">Fishermen sea-area warnings</div>
  <div class="box"><b>CAP 1.2 XML</b> (SACHET format) + SMS EN/HI/GU</div>
 </div>
 <div class="arrow">➜</div>
 <div class="col c5"><h3>5 · Backend</h3>
  <div class="box"><b>FastAPI</b> REST · OpenAPI /docs</div>
  <div class="box">PyTorch · LightGBM inference</div>
  <div class="box">SQLite audit log of every prediction</div>
  <div class="box">Scene cache · Gemini bulletin (guard-railed)</div>
  <div class="box">Docker image</div>
 </div>
 <div class="arrow">➜</div>
 <div class="col c6"><h3>6 · Dashboard</h3>
  <div class="box"><b>Operations</b> console: map, cone, RI, landfall, alerts</div>
  <div class="box"><b>Basin scan</b>: auto-detection & tracks</div>
  <div class="box"><b>Satellite</b>: Grad-CAM, Dvorak</div>
  <div class="box"><b>Validation</b>: all held-out metrics</div>
  <div class="box small">Next.js · Mapbox · Recharts · neumorphic UI</div>
 </div>
</div>""")

# 2. End-to-end flow ----------------------------------------------------------------------------
steps = [
    ("1", "Satellite scene", "Whole Bay of Bengal + Arabian Sea IR/WV image, every 3 h", "#2f6fd0"),
    ("2", "Identify", "Detector finds Ockhi as a depression near Sri Lanka", "#3a9b5c"),
    ("3", "Classify", "CNN: 28 kt, T2.0 → IMD Depression; Grad-CAM shows why", "#d49400"),
    ("4", "Warn early", "RI model: 3.5× normal risk of +30 kt in 24 h (Biparjoy)", "#e5484d"),
    ("5", "Predict", "Track + intensity to 72 h with a calibrated 67% cone", "#3b82c4"),
    ("6", "Assess impact", "Landfall window, 21 M people exposed, sea areas", "#8a5cd6"),
    ("7", "Alert", "12 Gujarat districts staged; CAP-XML + SMS in 3 languages", "#1f3f7a"),
]
body = "<h1>From one satellite image to one actionable alert</h1><div class='sub'>What happens inside the system — with real values from the Ockhi 2017 and Biparjoy 2023 backtests</div>"
body += "<div class='row' style='margin-top:70px;gap:10px'>"
for i, (n, t, d, c) in enumerate(steps):
    body += f"""<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:14px">
      <div style="width:92px;height:92px;border-radius:50%;background:{c};color:#fff;font-size:44px;font-weight:800;display:grid;place-items:center">{n}</div>
      <div style="font-size:25px;font-weight:800;color:{c};text-align:center;min-height:66px;display:flex;align-items:center">{t}</div>
      <div style="font-size:20px;text-align:center;line-height:1.38;background:#f5f7fb;border-radius:14px;padding:16px 10px;height:215px;border:1.5px solid #dde3ec">{d}</div></div>"""
    if i < len(steps) - 1:
        body += "<div class='arrow' style='margin-top:-150px'>➜</div>"
body += "</div>"
body += """<div style="margin-top:64px;display:flex;gap:18px">
 <div class="box" style="flex:1;font-size:19px"><b>Identification:</b> every demo storm found from raw imagery <b>24–42 h before</b> it became a Cyclonic Storm.</div>
 <div class="box" style="flex:1;font-size:19px"><b>Prediction:</b> uses only data available at that moment — every screen can be replayed against what really happened.</div>
 <div class="box" style="flex:1;font-size:19px"><b>Action:</b> output is a standard CAP 1.2 alert that alerting systems such as NDMA SACHET ingest (illustrative, status = Exercise).</div>
</div>"""
page("02_end_to_end_flow", body)

# 3. Training & validation ---------------------------------------------------------------------
page("03_training_validation", """
<h1>Training & validation — built so judges can trust the numbers</h1>
<div class="sub">Every model is scored on storms or seasons it never saw; splits are by storm and year, never random frames</div>
<table class="tall" style="margin-top:28px;font-size:19px">
<tr style="background:#1f3f7a;color:#fff"><th style="padding:12px;text-align:left">Model</th><th style="text-align:left">Training data</th><th style="text-align:left">Tuned on</th><th style="text-align:left">Tested on (never seen)</th><th style="text-align:left">Where trained</th></tr>
<tr style="background:#f5f7fb"><td style="padding:11px"><b>Intensity CNN</b> + Dvorak head</td><td>TCIR: Pacific & S. Hemisphere (pretrain) → NIO ≤2014</td><td>NIO 2015</td><td><b>NIO 2016</b> (Vardah, Kyant, Nada, Roanu) + 132 INSAT-3D images</td><td>Colab T4 GPU</td></tr>
<tr><td style="padding:11px"><b>Multi-channel CNN</b> IR/WV/PMW</td><td>same, with random channel dropout</td><td>NIO 2015</td><td><b>NIO 2016</b></td><td>Colab T4 GPU</td></tr>
<tr style="background:#f5f7fb"><td style="padding:11px"><b>Basin detector</b></td><td>339 GridSat scenes 2005–15 (+ storm-free scenes)</td><td>2016</td><td><b>189 scenes 2017–24</b> + 6 demo storms</td><td>Colab T4 GPU</td></tr>
<tr><td style="padding:11px"><b>ConvLSTM</b> sequences</td><td>TCIR 8-frame sequences, all basins</td><td>held-out storms</td><td><b>NIO 2016</b> + all-basin validation</td><td>Colab T4 GPU</td></tr>
<tr style="background:#f5f7fb"><td style="padding:11px"><b>RI early warning</b></td><td>IBTrACS NIO 1982–2016 (15:1 imbalance)</td><td>2017–19</td><td><b>2020–25</b> (Amphan, Tauktae, Mocha, Biparjoy…)</td><td>Laptop CPU</td></tr>
<tr><td style="padding:11px"><b>Genesis</b></td><td>IBTrACS depressions 1982–2016 + ERA5</td><td>2017–19</td><td><b>2020–25</b></td><td>Laptop CPU</td></tr>
<tr style="background:#f5f7fb"><td style="padding:11px"><b>Track + cone</b></td><td>IBTrACS NIO 1982–2016</td><td>2017–19 (cone calibrated here)</td><td><b>2020–25</b></td><td>Laptop CPU</td></tr>
</table>
<div class="row" style="margin-top:26px;gap:16px">
 <div class="box" style="flex:1;font-size:18px"><b>Class imbalance</b> handled explicitly: RI is 1 in 15 fixes → class-weighted training + probability calibration; threshold chosen on validation years only.</div>
 <div class="box" style="flex:1;font-size:18px"><b>Cross-basin transfer</b> for data-scarce NIO: pretraining on other ocean basins cut intensity error by <b>14.9%</b>.</div>
 <div class="box" style="flex:1;font-size:18px"><b>No look-ahead:</b> environment uses the previous month's ERA5; every replayed forecast uses data up to that time only.</div>
</div>""")

# 4. Results ------------------------------------------------------------------------------------
cards = [
    ("Identified before cyclone stage", "24–42 h", "6 of 6 demo storms, raw imagery only", "#3a9b5c"),
    ("Storm fixes detected", "79–94%", "centre error 41–85 km", "#3a9b5c"),
    ("Satellite wind error", "6.2–7.4 kt", "MAE on unseen 2016 NIO storms", "#2f6fd0"),
    ("Cross-basin transfer gain", "+14.9%", "vs training on NIO alone", "#2f6fd0"),
    ("RI storms warned (2020–25)", "8 / 8", "ROC-AUC 0.85 · 7 of 19 non-RI storms also flagged", "#e5484d"),
    ("Genesis skill", "AUC 0.89", "0.76 from a system's very first fix", "#d49400"),
    ("Uncertainty cone coverage", "66%", "target 67% — calibrated", "#3b82c4"),
    ("Track error 24 / 48 / 72 h", "140 / 300 / 447 km", "12–18% better than persistence", "#3b82c4"),
]
body = "<h1>Results on held-out data</h1><div class='sub'>No single 'accuracy %' — each model is measured the way forecasters measure it</div><div style='display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-top:26px'>"
for t, v, s, c in cards:
    body += f"""<div style="border-radius:18px;border:2.5px solid {c};padding:20px 18px;height:285px;background:#fbfcfe">
      <div style="font-size:18px;font-weight:700;color:#5b6679;text-transform:uppercase;letter-spacing:.4px">{t}</div>
      <div style="font-size:{52 if len(v) < 10 else 34}px;font-weight:800;color:{c};margin:14px 0 8px">{v}</div>
      <div style="font-size:18px;color:#3b4557;line-height:1.3">{s}</div></div>"""
body += "</div><div class='box' style='margin-top:20px;font-size:18px'><b>Stated honestly:</b> track error is ≈2× IMD's operational forecasts (no NWP steering winds) · detector precision 57% / recall 51% on all systems incl. weak depressions · extra satellite channels add ≈0.1 kt · image sequences did not improve RI — reported, not hidden.</div>"
page("04_results", body)

# 5. Differentiation ----------------------------------------------------------------------------
rows = [
    ("Identification", "Runs on a pre-cropped image of a known storm", "Scans the whole basin, finds & tracks systems automatically"),
    ("Early warning", "Class label after the storm exists", "Genesis probability + rapid-intensification warning with reasons"),
    ("Intensity", "Bare CNN class", "Dvorak T-number + IMD class, 3 cross-checked views, Grad-CAM"),
    ("Track", "Single deterministic line", "Calibrated 67% uncertainty cone (checked: 66%)"),
    ("Data scarcity", "Train on NIO only", "Cross-basin pretraining: +14.9%"),
    ("Impact", "Map only", "Landfall window, people exposed, fishermen sea areas"),
    ("Alerting", "Generic SMS mock", "CAP 1.2 XML (SACHET format) + EN/HI/GU SMS with segment counts"),
    ("Proof", "Live demo on chosen inputs", "Replay any storm since 1990 against what actually happened"),
    ("Honesty", "Accuracy % on random split", "Storm/year splits, false alarms & limits shown on screen"),
]
body = "<h1>What we built differently</h1><div class='sub'>Compared with the common PS 26070 pattern (CNN detects → LSTM predicts → map)</div><table style='margin-top:22px;font-size:21px'>"
body += "<tr><th style='width:18%'></th><th style='text-align:left;padding:12px;background:#eef0f4;color:#5b6679;border-radius:12px 0 0 0'>Typical build</th><th style='text-align:left;padding:12px;background:#1f3f7a;color:#fff;border-radius:0 12px 0 0'>Our prototype</th></tr>"
for i, (k, a, b) in enumerate(rows):
    bg = "#f7f8fb" if i % 2 == 0 else "#fff"
    body += f"<tr style='background:{bg}'><td style='padding:17px 12px;font-weight:800;color:#1f3f7a'>{k}</td><td style='padding:17px 12px;color:#6b7589'>✗ {a}</td><td style='padding:17px 12px;font-weight:600'>✓ {b}</td></tr>"
body += "</table>"
page("05_differentiation", body)

# 6. Deployment ---------------------------------------------------------------------------------
page("06_deployment", """
<h1>Deployment — how it goes live</h1>
<div class="sub">Free-tier friendly: static-quality frontend on Vercel, ML backend as a Docker container on Hugging Face Spaces</div>
<div class="row" style="margin-top:34px;height:560px;gap:22px">
 <div class="col c6" style="flex:.8"><h3>Users</h3>
  <div class="box">RSMC forecaster</div><div class="box">SDMA / NDRF</div><div class="box">District collector</div><div class="box">Judges (public link)</div></div>
 <div class="arrow">➜</div>
 <div class="col c1"><h3>Vercel</h3>
  <div class="box"><b>Next.js dashboard</b><br>global CDN, HTTPS</div>
  <div class="box">Env: NEXT_PUBLIC_API_URL, NEXT_PUBLIC_MAPBOX_TOKEN</div>
  <div class="box small">auto-deploys on every git push</div></div>
 <div class="arrow">➜</div>
 <div class="col c3" style="flex:1.3"><h3>Hugging Face Space (Docker)</h3>
  <div class="box"><b>FastAPI + PyTorch + LightGBM</b><br>free CPU tier: 2 vCPU · 16 GB RAM</div>
  <div class="box">Image contains models (≈330 MB) + processed data</div>
  <div class="box">Secrets: GEMINI_API_KEY · CORS_ORIGINS</div>
  <div class="box small">port 7860 · /api/health · /docs</div></div>
 <div class="arrow">➜</div>
 <div class="col c2"><h3>External services</h3>
  <div class="box">Mapbox base maps</div><div class="box">Gemini (optional bulletin)</div><div class="box">MOSDAC / GridSat feed (future live ingest)</div></div>
</div>
<div class="row" style="margin-top:26px;gap:18px">
 <div class="box" style="flex:1;font-size:18px"><b>Training path:</b> Google Colab (T4 GPU) → Google Drive → models/weights → Docker image</div>
 <div class="box" style="flex:1;font-size:18px"><b>Code path:</b> GitHub repo → Vercel (frontend) + HF Space (backend) · alternatives: Render, Railway, Cloud Run</div>
 <div class="box" style="flex:1;font-size:18px"><b>Local / offline demo:</b> run_dev.ps1 or docker run — works without internet except map tiles</div>
</div>""")

# 7. Users & decisions --------------------------------------------------------------------------
users = [
    ("RSMC forecaster", "Is it forming? Will it intensify rapidly? Where will it go?", "Basin scan, genesis, RI warning + reasons, Dvorak/Grad-CAM, cone", "#2f6fd0"),
    ("SDMA / NDRF", "Where to pre-position teams, and when?", "Landfall window, people exposed, district staging", "#e5484d"),
    ("District collector", "Evacuate now, prepare, or watch?", "Red/Orange/Yellow per district with ETA; CAP alert", "#d49400"),
    ("Fisheries & fisherfolk", "Can boats go out? Must they return?", "Sea-area warnings; SMS in Hindi/Gujarati", "#3a9b5c"),
]
body = "<h1>Who uses it — and the decision each screen supports</h1><div class='sub'>Every panel on the dashboard is labelled with its intended user</div><div style='display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin-top:34px'>"
for u, q, a, c in users:
    body += f"""<div style="border-radius:20px;border:3px solid {c};overflow:hidden;height:660px;display:flex;flex-direction:column">
      <div style="background:{c};color:#fff;font-size:27px;font-weight:800;padding:22px 18px;text-align:center">{u}</div>
      <div style="padding:24px 20px;font-size:25px;line-height:1.35"><div style="font-size:16px;font-weight:800;color:#7a8597;text-transform:uppercase;margin-bottom:8px">Decision</div>{q}</div>
      <div style="padding:0 20px 22px;font-size:21px;line-height:1.35;margin-top:auto"><div style="font-size:16px;font-weight:800;color:#7a8597;text-transform:uppercase;margin-bottom:8px">What the system gives</div><b>{a}</b></div></div>"""
body += "</div>"
page("07_users_decisions", body)

# 8. Risks & mitigation -------------------------------------------------------------------------
risks = [
    ("Track skill below NWP", "Position as decision support; show calibrated cone + 'what actually happened' replay; add GFS steering winds next"),
    ("Sensor / processing shift (INSAT vs training)", "Measured (≈15 kt on INSAT JPEGs); linear calibration; retrain on native MOSDAC HDF5"),
    ("Missing channels (e.g. no WV in 2024)", "Models trained with channel dropout; IR-only fallback; IR→WV estimate flagged in UI"),
    ("False alarms (RI, detector)", "Shown on screen; thresholds tuned on validation years; tracks confirmed after 15 h"),
    ("Rare extreme storms (ESCS/SuCS)", "Cross-basin pretraining; flagged as untested class; Amphan demo covers SuCS"),
    ("Over-trust / impersonation", "CAP status=Exercise; LLM bulletin constrained to model numbers and stripped of agency headers"),
]
body = "<h1>Risks we found — and how the prototype handles them</h1><div class='sub'>All discovered while building and testing, not hypothetical</div><table style='margin-top:26px;font-size:21px'>"
for i, (r, m) in enumerate(risks):
    body += f"<tr style='background:{'#fff5f5' if i % 2 == 0 else '#fff'}'><td style='padding:30px 14px;width:34%;font-weight:800;color:#c0392b'>⚠ {r}</td><td style='padding:30px 14px'>→ {m}</td></tr>"
body += "</table>"
page("08_risks_mitigation", body)

# 9. Video timeline -----------------------------------------------------------------------------
segs = [("0:00", "Hook", 30, "#e5484d"), ("0:30", "Problem & gap", 30, "#d49400"), ("1:00", "Identify: Basin scan", 50, "#3a9b5c"),
        ("1:50", "Classify: Satellite", 40, "#2f6fd0"), ("2:30", "Warn & predict: Operations", 60, "#3b82c4"),
        ("3:30", "Impact & alerts", 40, "#8a5cd6"), ("4:10", "Validation", 30, "#1f3f7a"), ("4:40", "Close", 20, "#5b6679")]
body = "<h1>5-minute demo video — timeline</h1><div class='sub'>Each block = one screen; full voice-over script is in the guide</div><div style='display:flex;margin-top:120px;height:190px;border-radius:18px;overflow:hidden'>"
for t, n, s, c in segs:
    body += f"<div style='flex:{s};background:{c};color:#fff;padding:16px 12px;border-right:3px solid #fff'><div style='font-size:22px;font-weight:800'>{t}</div><div style='font-size:22px;font-weight:700;margin-top:14px;line-height:1.2'>{n}</div><div style='font-size:17px;margin-top:10px;opacity:.9'>{s} s</div></div>"
body += "</div><div style='display:flex;justify-content:space-between;font-size:20px;color:#5b6679;margin-top:10px'><span>0:00</span><span>1:00</span><span>2:00</span><span>3:00</span><span>4:00</span><span>5:00</span></div>"
body += "<div class='row' style='margin-top:70px;gap:18px'><div class='box' style='flex:1;font-size:20px'>Record at <b>1600×900</b>, browser zoom 100%, dark mode off.</div><div class='box' style='flex:1;font-size:20px'>Use the <b>deep links</b> in the guide to jump to each moment.</div><div class='box' style='flex:1;font-size:20px'>Keep the backend warm: open each page once before recording.</div></div>"
page("09_video_timeline", body)
