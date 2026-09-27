import { Card } from "@/components/ui";

const PIPELINE = [
  ["Data", "INSAT-3D (Kaggle, MOSDAC-sourced) · TCIR IR benchmark (IO + CPAC + SH) · IBTrACS best tracks 1982–2025 · ERA5 SST, shear, humidity"],
  ["Satellite model", "EfficientNet-B0 on IR brightness temperature → wind, Dvorak T-number and IMD class heads; cross-basin pretraining; Grad-CAM"],
  ["RI early warning", "LightGBM, class-weighted for 15:1 imbalance, Platt-calibrated; SHAP explanations per forecast"],
  ["Genesis", "LightGBM: will a depression become a Cyclonic Storm within 72 h?"],
  ["Track & intensity", "LSTM over the last 18 h; Monte-Carlo dropout spread calibrated to a 67% cone"],
  ["Dissemination", "District staging (watch / alert / warning) → CAP 1.2 XML in English, Hindi, Gujarati; optional LLM bulletin constrained to model numbers"],
  ["Stack", "FastAPI + SQLite audit log · Next.js + Mapbox GL + Recharts · Docker Compose"],
];

const LIMITS = [
  "Decision support only — not a replacement for IMD / RSMC New Delhi bulletins. Not connected to SACHET; CAP output is marked status=Exercise.",
  "Track errors are roughly twice IMD's operational errors because we use no numerical-weather-prediction steering flow. Skill drops sharply beyond 48 h.",
  "The satellite model was trained on TCIR imagery; on INSAT-3D JPEGs its error doubles (≈15 kt after calibration). Production use needs training on native MOSDAC HDF5 data.",
  "No Extremely Severe or Super Cyclonic Storm frames in the satellite test set, so skill for the strongest storms is unproven.",
  "RI is rare (≈1 in 15 fixes). At our threshold, 7 of 19 non-RI test storms were also flagged — the false-alarm cost (evacuation fatigue, fishing bans) must be weighed when setting the threshold.",
  "ERA5 environment uses previous-month means (no look-ahead), which is too coarse to capture shear changes; daily fields are the next step.",
  "District points and gale radii are illustrative approximations, not official boundaries or wind-radii analyses.",
];

const NEXT = [
  "Train on native INSAT-3D/3DR/3DS HDF5 from MOSDAC and ingest live imagery at operational cadence.",
  "Add GFS / ERA5 steering flow and daily shear to the track and RI models.",
  "Pilot the CAP output against NDMA/C-DOT's real SACHET ingestion; add Odia, Tamil, Telugu, Bengali, Marathi.",
  "Extend to the full RSMC New Delhi area (Bangladesh, Myanmar, Sri Lanka, Pakistan, Oman).",
  "Physics-informed generative augmentation for rare extreme storms; foundation-model track forecasting.",
];

export default function About() {
  return (
    <div className="grid gap-6 xl:grid-cols-12">
      <Card className="xl:col-span-7" title="What this prototype does" subtitle="SIH 2026 · PS 26070 · Ministry of Earth Sciences / IMD · Disaster Management">
        <div className="space-y-3">
          {PIPELINE.map(([k, v], i) => (
            <div key={k} className="neu-inset-sm flex gap-4 px-4 py-3">
              <div className="neu-sm grid h-8 w-8 shrink-0 place-items-center text-xs font-extrabold text-accent">{i + 1}</div>
              <div><div className="text-sm font-bold text-ink">{k}</div><div className="text-xs text-muted">{v}</div></div>
            </div>
          ))}
        </div>
      </Card>
      <div className="space-y-6 xl:col-span-5">
        <Card title="Known limitations" subtitle="Stated up front, because false confidence costs lives">
          <ul className="space-y-2.5">
            {LIMITS.map((l) => <li key={l} className="neu-inset-sm px-4 py-2.5 text-xs text-muted">{l}</li>)}
          </ul>
        </Card>
        <Card title="Next steps">
          <ul className="list-disc space-y-1.5 pl-5 text-xs text-muted">{NEXT.map((n) => <li key={n}>{n}</li>)}</ul>
        </Card>
      </div>
    </div>
  );
}
