# Cyclone Decision Support — SIH 2026 · PS 26070

AI-assisted tropical-cyclone guidance for the North Indian Ocean: satellite intensity estimation (Dvorak-grounded, with Grad-CAM), rapid-intensification early warning, cyclogenesis outlook, track forecast with a calibrated uncertainty cone, and district alerts as CAP 1.2 XML (SACHET-compatible format, illustrative).

> Research prototype. Not an official IMD / NDMA product and not connected to SACHET.

## Run it

```powershell
# one-time setup
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd frontend; npm install; cd ..

# every time
.\run_dev.ps1          # API on :8000 (docs at /docs), dashboard on :3000
```

Keys live in `.env` (copy `.env.example`). The dashboard needs `frontend/.env.local` with `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_MAPBOX_TOKEN`. Without a Gemini key, bulletins fall back to a template.

## Layout

| Path | What |
|---|---|
| `ml/` | Shared code: IMD scale + Dvorak table, ADT-lite IR features, IBTrACS features, ERA5 environment, RI / genesis / track trainers |
| `notebooks/train_intensity_colab.ipynb` | GPU training for the satellite model (built from `notebooks/src/` by `notebooks/build_notebook.py`) |
| `notebooks/train_advanced_colab.ipynb` | Multi-channel model, ConvLSTM, GridSat basin detector, demo scenes (`python notebooks/build_notebook.py train_advanced_colab`) |
| `models/weights/` | Trained models + `*_metrics.json` |
| `backend/app/` | FastAPI: analysis, Grad-CAM, CAP-XML alerts, bulletins, SQLite audit log |
| `frontend/` | Next.js neumorphic dashboard (Operations, Satellite analysis, Validation, About) |
| `scripts/fetch_era5_monthly.py` | ERA5 SST / shear / humidity download (Copernicus CDS key) |

## Retrain

```powershell
.venv\Scripts\python -m ml.train_ri        # rapid intensification
.venv\Scripts\python -m ml.train_genesis   # cyclogenesis
.venv\Scripts\python -m ml.train_track     # track + cone
# satellite model: run the notebook on Colab (T4), copy outputs to models/weights/
```

## Results on held-out data

| Model | Test data | Result |
|---|---|---|
| Satellite intensity | NIO storms 2016 (Vardah, Kyant, Nada, Roanu…) | MAE 7.4 kt, RMSE 10.3 kt; cross-basin pretraining +14.9% vs NIO-only |
| Rapid intensification | Seasons 2020–25 | ROC-AUC 0.85, PR-AUC 0.44 (base rate 0.10); all 8 RI storms flagged, 7 of 19 non-RI storms falsely flagged |
| Track | Seasons 2020–25 | 140 / 300 / 447 km at 24 / 48 / 72 h (12–18% better than persistence, ~2× IMD operational error); cone coverage 66% vs 67% target |
| Genesis | Seasons 2020–25 | ROC-AUC 0.89 (0.76 from a system's first fix) |
| Full-basin detector (GridSat IR+WV) | 189 scenes, 2017–24 | Precision 57%, recall 51% (all systems ≥17 kt), centre error 69 km |
| Image-only pipeline, 6 demo storms | Vardah, Ockhi, Fani, Amphan, Tauktae, Dana | Detected 24–42 h before Cyclonic Storm stage; 79–94% of fixes; centre error 41–85 km; 14 confirmed real vs 6 confirmed false tracks |
| Multi-channel intensity (IR/WV/PMW) | NIO 2016 | MAE 6.2–6.5 kt; extra channels change error by only ~0.1 kt |
| ConvLSTM image sequence | All-basin validation | 24 h change MAE 13.3 kt (vs 13.9 without images); RI PR-AUC worse with images (0.30 vs 0.42) — shown as experimental |

Limitations are listed on the dashboard's About page and in `models/weights/*_metrics.json`.
