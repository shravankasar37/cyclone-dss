"use client";
import { useEffect, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Upload } from "lucide-react";
import { API, CLASS_COLORS, STATIC_MODE, fmtTime, get, type ClassCode, type ImageAnalysis, type SatFrame, type SatStorm } from "@/lib/api";
import { usePalette } from "@/lib/theme";
import { dayTicks } from "@/components/Charts";
import { ClassPill, ErrorBox, Loading, Segmented } from "@/components/ui";
import { Mini, Panel } from "@/components/Console";

type View = "ir" | "gradcam";
interface Series { k: number; time: string; best_track_kt: number; cnn_kt: number; dvorak_kt: number }
const SCENE_TEXT: Record<string, string> = {
  EYE: "Eye pattern — warm eye surrounded by a cold, closed ring of convection",
  CDO: "Central Dense Overcast — large, symmetric cold cloud mass over the centre",
  CURVED_BAND: "Curved band — cold spiral band wrapping around the centre",
  SHEAR: "Shear pattern — convection displaced to one side of the centre",
  DISORGANIZED: "Disorganised convection — no clear Dvorak pattern yet",
};

export default function Satellite() {
  const [storms, setStorms] = useState<SatStorm[]>([]);
  const [sid, setSid] = useState<string | null>(null);
  const [frames, setFrames] = useState<SatFrame[]>([]);
  const [series, setSeries] = useState<Series[]>([]);
  const [i, setI] = useState(0);
  const [res, setRes] = useState<ImageAnalysis | null>(null);
  const [view, setView] = useState<View>("gradcam");
  const [err, setErr] = useState<string | null>(null);
  const [upload, setUpload] = useState<ImageAnalysis | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const cache = useRef(new Map<number, ImageAnalysis>());
  const p = usePalette();

  useEffect(() => {
    get<SatStorm[]>("/api/satellite/storms").then((s) => { setStorms(s); setSid(s.find((x) => x.name === "VARDAH")?.tcir_sid ?? s[0]?.tcir_sid ?? null); })
      .catch((e) => setErr(String(e.message ?? e)));
  }, []);
  useEffect(() => {
    if (!sid) return;
    Promise.all([get<SatFrame[]>(`/api/satellite/frames?tcir_sid=${sid}`), get<Series[]>(`/api/satellite/series?tcir_sid=${sid}`)]).then(([f, s]) => {
      setFrames(f); setSeries(s);
      setI(f.reduce((b, x, j) => (x.best_track_kt > f[b].best_track_kt ? j : b), 0));
    });
  }, [sid]);
  const k = frames[i]?.k;
  useEffect(() => {
    if (k == null) return;
    const hit = cache.current.get(k);
    if (hit) { setRes(hit); return; }
    const h = setTimeout(() => get<ImageAnalysis>(`/api/satellite/frames/${k}`).then((r) => { cache.current.set(k, r); setRes(r); }), 150);
    return () => clearTimeout(h);
  }, [k]);

  const onFile = async (f: File) => {
    setUploading(true);
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await fetch(`${API}/api/satellite/upload?insat_calibration=true`, { method: "POST", body: fd });
      setUpload(await r.json());
    } finally { setUploading(false); }
  };

  if (err) return <ErrorBox msg={err} />;
  const storm = storms.find((s) => s.tcir_sid === sid);
  const ms = (iso: string) => Date.parse(iso + "Z");

  return (
    <div className="console-h flex flex-col gap-4">
      <section className="neu flex shrink-0 flex-wrap items-center gap-2 px-4 py-2.5">
        <span className="label mr-1">TCIR IR</span>
        {storms.map((s) => (
          <button key={s.tcir_sid} data-active={s.tcir_sid === sid} onClick={() => setSid(s.tcir_sid)} className="neu-btn px-2.5 py-1.5 text-[11px] font-bold">
            {s.name === "UNNAMED" ? s.tcir_sid.slice(0, 6) : s.name[0] + s.name.slice(1).toLowerCase()} <span className="font-medium text-faint">{s.start.slice(2, 4)}</span>
            {s.held_out && <span className="ml-1 text-[9px] font-bold text-sev-green">●</span>}
          </button>
        ))}
        <div className="min-w-[160px] flex-1 px-2">
          <input type="range" className="neu-range" min={0} max={Math.max(0, frames.length - 1)} value={i} onChange={(e) => setI(+e.target.value)} />
        </div>
        <div className="neu-inset-sm px-3 py-1.5 text-center">
          <div className="text-[12px] font-bold tabular text-ink">{frames[i] ? fmtTime(frames[i].time) : "—"}</div>
          <div className="text-[10px] text-faint">frame {i + 1}/{frames.length} · <span className="text-sev-green">●</span> = never seen in training</div>
        </div>
      </section>

      <div className="grid min-h-0 flex-1 gap-4 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.1fr)_minmax(0,1fr)]">
        <Panel title={storm ? `${storm.name === "UNNAMED" ? storm.tcir_sid : storm.name[0] + storm.name.slice(1).toLowerCase()} · ${frames[i] ? `${frames[i].lat.toFixed(1)}°N ${frames[i].lon.toFixed(1)}°E` : ""}` : "Satellite frame"}
          right={<Segmented<View> value={view} onChange={setView} options={[{ value: "ir", label: "IR" }, { value: "gradcam", label: "Grad-CAM" }]} />}>
          <div className="flex h-full min-h-[300px] flex-col">
            <div className="neu-inset mx-auto aspect-square w-full min-h-0 overflow-hidden p-2" style={{ maxWidth: "min(100%, calc(100dvh - 17rem))" }}>
              {res ? <img src={view === "ir" ? res.image_png : res.gradcam_png} alt="IR satellite frame" className="h-full w-full rounded-[14px] object-cover" /> : <Loading h={300} />}
            </div>
            <p className="mt-2 shrink-0 text-center text-[10px] text-faint">Grad-CAM shows which cloud regions drove the wind estimate. Cold (high) cloud tops are bright.</p>
          </div>
        </Panel>

        <Panel title="Dvorak-grounded intensity" bodyClass="overflow-y-auto pr-1">
          {res ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <Mini label="CNN wind" value={<>{Math.round(res.cnn_wind_kt)} <span className="text-[10px] text-muted">kt</span></>} sub="± 7.4 kt typical (test MAE)" />
                <Mini label="Dvorak T-number" value={`T${res.t_number.toFixed(1)}`} sub={`→ ${Math.round(res.dvorak_wind_kt)} kt via Dvorak table`} />
                <Mini label="Best track (truth)" value={res.best_track_kt != null ? <>{Math.round(res.best_track_kt)} <span className="text-[10px] text-muted">kt</span></> : "—"} sub="JTWC 1-min" />
                <Mini label="Pressure (est.)" value={<>{Math.round(res.pressure_est_hpa)} <span className="text-[10px] text-muted">hPa</span></>} sub="wind–pressure relation" />
              </div>
              {res.seq && (
                <div className="neu-inset-sm px-3 py-2 text-[11px]">
                  <div className="label mb-1">ConvLSTM · next 24 h from last 21 h of images</div>
                  <span className="text-ink"><b className="tabular">{res.seq.dv24_kt >= 0 ? "+" : ""}{Math.round(res.seq.dv24_kt)} kt</b></span>
                  <span className="ml-3 text-ink"><b className="tabular">{Math.round(res.seq.ri_probability * 100)}%</b> <span className="text-faint">RI risk</span></span>
                  {res.seq.observed_dv24_kt != null && <span className="ml-3 text-muted">actual {res.seq.observed_dv24_kt >= 0 ? "+" : ""}{Math.round(res.seq.observed_dv24_kt)} kt</span>}
                </div>
              )}
              <div className="grid grid-cols-2 gap-2">
                <div className="neu-inset-sm px-3 py-2">
                  <div className="label mb-1.5">IMD class — 3 views</div>
                  <div className="space-y-1.5 text-[11px]">
                    {[["Classifier", res.class_head.code], ["CNN wind", res.class_from_wind.code], ["Dvorak T", res.class_from_dvorak.code]].map(([l, c]) => (
                      <div key={l} className="flex items-center justify-between"><span className="text-muted">{l}</span><ClassPill code={c as ClassCode} size="sm" /></div>
                    ))}
                  </div>
                  <div className="mt-2 text-[10px] font-semibold" style={{ color: res.class_from_wind.code === res.class_from_dvorak.code ? "var(--green)" : "var(--orange)" }}>
                    {res.class_from_wind.code === res.class_from_dvorak.code ? "Agree → higher confidence" : "Disagree → use caution"}
                  </div>
                </div>
                <div className="neu-inset-sm px-3 py-2">
                  <div className="label mb-1">Scene type (ADT-lite)</div>
                  <div className="text-[13px] font-bold text-ink">{res.scene_type.replace("_", " ")}</div>
                  <div className="text-[10px] leading-snug text-muted">{SCENE_TEXT[res.scene_type]}</div>
                  <div className="mt-1.5 grid grid-cols-2 gap-x-2 text-[10px] tabular">
                    <span className="text-faint">Eye</span><span className="text-ink">{(res.adt_features.eye_bt - 273.15).toFixed(0)} °C</span>
                    <span className="text-faint">Ring</span><span className="text-ink">{(res.adt_features.cloud_bt - 273.15).toFixed(0)} °C</span>
                    <span className="text-faint">&lt;−60 °C</span><span className="text-ink">{Math.round(res.adt_features.cold_frac_60 * 100)}%</span>
                  </div>
                </div>
              </div>
              <div>
                <div className="label mb-1.5">Class probabilities</div>
                <div className="flex gap-1.5">
                  {(Object.entries(res.class_probs) as [ClassCode, number][]).map(([c, v]) => (
                    <div key={c} className="flex-1 text-center">
                      <div className="neu-inset-sm relative mx-auto h-12 w-full overflow-hidden">
                        <div className="absolute inset-x-1 bottom-1 rounded-md" style={{ height: `${Math.max(3, v * 100) * 0.4}px`, background: CLASS_COLORS[c] }} />
                      </div>
                      <div className="mt-0.5 text-[9.5px] font-bold text-muted">{c}</div>
                      <div className="text-[9.5px] tabular text-faint">{Math.round(v * 100)}%</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : <Loading h={400} />}
        </Panel>

        <div className="flex min-h-0 flex-col gap-4">
          <Panel title="Whole-storm backtest (kt)" className="min-h-[220px] flex-1">
            <div className="h-full">
              {p && series.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={series.map((s) => ({ ...s, t: ms(s.time) }))} margin={{ top: 6, right: 8, left: -22, bottom: 0 }}>
                    <CartesianGrid stroke={p["--sd"]} strokeOpacity={0.5} vertical={false} />
                    <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} ticks={dayTicks(series.map((s) => ms(s.time)))} stroke={p["--faint"]} fontSize={10} tickLine={false}
                      tickFormatter={(t) => new Date(t).toLocaleString("en-GB", { day: "2-digit", month: "short", timeZone: "UTC" })} />
                    <YAxis stroke={p["--faint"]} fontSize={10} tickLine={false} />
                    <Tooltip contentStyle={{ background: p["--bg"], border: "none", borderRadius: 14, fontSize: 12 }}
                      labelFormatter={(t) => new Date(t as number).toUTCString().slice(5, 22)} formatter={(v, n) => [`${Math.round(v as number)} kt`, n]} />
                    <Line dataKey="best_track_kt" name="best track" stroke={p["--muted"]} strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    <Line dataKey="cnn_kt" name="CNN" stroke={p["--accent"]} strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line dataKey="dvorak_kt" name="Dvorak (from T)" stroke={p["--orange"]} strokeWidth={1.5} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
                    {frames[i] && <ReferenceLine x={ms(frames[i].time)} stroke={p["--accent"]} />}
                  </LineChart>
                </ResponsiveContainer>
              ) : <Loading h={180} />}
            </div>
          </Panel>
          <Panel title="Try your own INSAT image" className="shrink-0">
            <div className="flex items-center gap-3">
              <button onClick={() => fileRef.current?.click()} disabled={STATIC_MODE} className="neu-btn grid h-24 w-24 shrink-0 place-items-center overflow-hidden text-[11px] font-semibold">
                {upload ? <img src={upload.gradcam_png} alt="uploaded" className="h-full w-full object-cover p-1.5" /> : <span className="flex flex-col items-center gap-1"><Upload size={20} />{uploading ? "Analysing…" : "Choose"}</span>}
              </button>
              <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
              {upload ? (
                <div className="grid flex-1 grid-cols-2 gap-2">
                  <Mini label="Calibrated wind" value={`${Math.round(upload.cnn_wind_kt)} kt`} sub={`raw ${Math.round(upload.cnn_wind_raw_kt)} kt`} />
                  <Mini label="Dvorak T / class" value={<span className="flex items-center gap-2">T{upload.t_number.toFixed(1)} <ClassPill code={upload.class_from_wind.code} size="sm" /></span>} sub={upload.scene_type.replace("_", " ")} />
                </div>
              ) : STATIC_MODE ? <p className="text-[11px] text-muted">Uploading your own image runs the model live, so it is available in the full (local / server) version. This online replay edition shows pre-computed results for every storm.</p>
              : <p className="text-[11px] text-muted">Storm-centred IR JPEG (cold = bright). Samples: <code className="font-mono text-[10px]">data/raw/insat3d/insat3d_raw_cyclone_ds/</code> — file name = true knots. INSAT error ≈15 kt (different processing).</p>}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
