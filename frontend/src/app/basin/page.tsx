"use client";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChevronLeft, ChevronRight, Pause, Play, Radar } from "lucide-react";
import { get, fmtTime, type BasinScene, type BasinStorm, type ClassCode } from "@/lib/api";
import { usePalette } from "@/lib/theme";
import { dayTicks } from "@/components/Charts";
import { Card, ClassPill, ErrorBox, Loading } from "@/components/ui";
import { MapLegend, Mini, Panel } from "@/components/Console";

const BasinMap = dynamic(() => import("@/components/BasinMap"), { ssr: false, loading: () => <Loading h={560} /> });
interface Status { available: boolean; storms: { storm: string; scenes: number; start: string; end: string }[]; missing?: string[] }
const title = (s: string) => s[0] + s.slice(1).toLowerCase();
const hrs = (h: number | null | undefined) => (h == null ? "—" : h > 0 ? `${Math.round(h)} h earlier` : h === 0 ? "same time" : `${Math.round(-h)} h later`);

export default function Basin() {
  const [status, setStatus] = useState<Status | null>(null);
  const [name, setName] = useState<string | null>(null);
  const [storm, setStorm] = useState<BasinStorm | null>(null);
  const [pos, setPos] = useState(0);
  const [scene, setScene] = useState<BasinScene | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const sceneCache = useRef(new Map<number, BasinScene>());
  const p = usePalette();

  useEffect(() => {
    get<Status>("/api/basin/storms").then((s) => {
      setStatus(s);
      const want = new URLSearchParams(window.location.search).get("storm")?.toUpperCase();  // deep link: /basin?storm=FANI
      if (s.available && s.storms.length) setName((s.storms.find((x) => x.storm === want) ?? s.storms.find((x) => x.storm === "OCKHI") ?? s.storms[0]).storm);
    }).catch((e) => setErr(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    if (!name) return;
    let live = true;
    get<BasinStorm>(`/api/basin/storms/${name}`, 2).then((s) => {
      if (!live) return;
      setStorm(s); setSelected(s.summary.main_track);
      const main = s.tracks.find((t) => t.id === s.summary.main_track);
      setPos(Math.max(0, main ? s.ks.indexOf(main.points[0].k) : 0));
    }).catch((e) => setErr(String(e.message ?? e)));
    return () => { live = false; };
  }, [name]);

  const k = storm?.ks[pos];
  useEffect(() => {
    if (k == null) return;
    const hit = sceneCache.current.get(k);
    if (hit) { setScene(hit); return; }
    let live = true;
    get<BasinScene>(`/api/basin/scene/${k}`).then((s) => { sceneCache.current.set(k, s); if (live) setScene(s); });
    return () => { live = false; };
  }, [k]);

  useEffect(() => {
    if (!playing || !storm) return;
    const h = setInterval(() => setPos((i) => { if (i >= storm.ks.length - 1) { setPlaying(false); return i; } return i + 1; }), 900);
    return () => clearInterval(h);
  }, [playing, storm]);

  const track = storm?.tracks.find((t) => t.id === selected) ?? null;
  const point = track?.points.find((q) => q.k === k) ?? null;
  const series = useMemo(() => {
    if (!storm) return [];
    const main = storm.tracks.find((t) => t.id === storm.summary.main_track);
    const rows = new Map<number, Record<string, number | null>>();
    const ms = (iso: string) => Date.parse(iso + (iso.endsWith("Z") ? "" : "Z"));
    (storm.summary.best_track ?? []).forEach((b) => rows.set(ms(b.time), { ...(rows.get(ms(b.time)) ?? {}), t: ms(b.time), best: Math.round(b.wind_imd_kt) }));
    main?.points.forEach((q) => rows.set(ms(q.time), { ...(rows.get(ms(q.time)) ?? {}), t: ms(q.time), image: Math.round(q.wind_imd_kt),
      ri: q.seq ? Math.round(q.seq.ri_probability * 100) : null }));
    return [...rows.values()].sort((a, b) => (a.t as number) - (b.t as number));
  }, [storm]);

  const selectStorm = (s: string) => { setPlaying(false); setStorm(null); setScene(null); setName(s); };

  if (err) return <ErrorBox msg={err} />;
  if (!status) return <Loading h={500} />;
  if (!status.available) {
    return (
      <Card title="Full-basin identification" subtitle="Needs the advanced models from the second Colab notebook">
        <p className="text-sm text-muted">Run <code className="font-mono text-xs">notebooks/train_advanced_colab.ipynb</code> on Colab and copy its outputs to <code className="font-mono text-xs">models/weights/</code>. Missing: {status.missing?.join(", ")}</p>
      </Card>
    );
  }
  const sm = storm?.summary;

  return (
    <div className="console-h flex flex-col gap-4">
      <section className="neu flex shrink-0 flex-wrap items-center gap-2 px-4 py-2.5">
        <span className="label mr-1 flex items-center gap-1.5"><Radar size={13} /> Basin scan</span>
        {status.storms.map((s) => (
          <button key={s.storm} data-active={s.storm === name} onClick={() => selectStorm(s.storm)} className="neu-btn px-2.5 py-1.5 text-[11px] font-bold">
            {title(s.storm)} <span className="font-medium text-faint">{s.start.slice(0, 4)}</span>
          </button>
        ))}
        {storm && (
          <>
            <div className="ml-2 flex items-center gap-2">
              <button className="neu-btn grid h-9 w-9 place-items-center" onClick={() => setPos((i) => Math.max(0, i - 1))} aria-label="Previous"><ChevronLeft size={15} /></button>
              <button className="neu-btn grid h-9 w-9 place-items-center text-accent" data-active={playing} aria-label="Play"
                onClick={() => { if (pos >= storm.ks.length - 1) setPos(0); setPlaying((x) => !x); }}>{playing ? <Pause size={15} /> : <Play size={15} />}</button>
              <button className="neu-btn grid h-9 w-9 place-items-center" onClick={() => setPos((i) => Math.min(storm.ks.length - 1, i + 1))} aria-label="Next"><ChevronRight size={15} /></button>
            </div>
            <div className="min-w-[160px] flex-1 px-2"><input type="range" className="neu-range" min={0} max={storm.ks.length - 1} value={pos} onChange={(e) => { setPlaying(false); setPos(+e.target.value); }} /></div>
            <div className="neu-inset-sm px-3 py-1.5 text-center text-[12px] font-bold tabular text-ink">{storm.times[pos] ? fmtTime(storm.times[pos]) : "—"}</div>
          </>
        )}
      </section>

      <div className="grid min-h-0 flex-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(340px,400px)]">
        <section className="neu flex min-h-[460px] flex-col p-2">
          <div className="neu-inset relative flex-1">
            {storm ? <BasinMap storm={storm} scene={scene} k={k ?? 0} selected={selected} onSelect={setSelected} />
              : <div className="shimmer grid h-full place-items-center text-sm text-muted">Scanning {status.storms.find((s) => s.storm === name)?.scenes ?? ""} satellite scenes…</div>}
            <MapLegend items={[
              [<span key="c" className="h-2 w-2 rounded-full bg-[#e5484d]" />, "Detected cyclone"],
              [<span key="d" className="h-2 w-2 rounded-full bg-[#e0a100]" />, "Detected depression"],
              [<span key="a" className="h-[3px] w-4 rounded bg-accent" />, "Automatic track"],
              [<span key="b" className="h-0 w-4 border-t-2 border-dashed border-[#6a778b]" />, "Best track (truth)"],
              [<span key="i" className="h-2.5 w-8 rounded-sm" style={{ background: "linear-gradient(90deg,#9aa0a8,#78beff,#2a6ef0,#28c85a,#fadc28,#f05a1e,#c81428)" }} />, "IR: warm → very cold cloud"],
            ]} />
          </div>
          <p className="shrink-0 px-2 pt-2 text-[10px] text-faint">Input is only the raw 3-hourly GridSat IR + water-vapour image of the whole basin — no best-track data is used for detection, tracking, intensity or RI.</p>
        </section>

        <div className="flex min-h-0 flex-col gap-4">
          <Panel title="How early and how accurate?" className="shrink-0">
            {sm ? (
              <div className="grid grid-cols-2 gap-2">
                <Mini label="First detected" value={<span className="text-[12px]">{sm.first_detection ? fmtTime(sm.first_detection) : "—"}</span>} sub={`vs depression: ${hrs(sm.hours_before_depression)}`} />
                <Mini label="Before cyclone stage" value={<span className="text-[13px]">{hrs(sm.hours_before_cyclonic_storm)}</span>} sub="than IMD CS (≥34 kt)" />
                <Mini label="Detected at" value={sm.detection_rate != null ? `${Math.round(sm.detection_rate * 100)}%` : "—"} sub="of best-track fixes" />
                <Mini label="Centre error" value={sm.median_position_error_km != null ? Math.round(sm.median_position_error_km) : "—"} unit="km" sub={sm.intensity_mae_kt != null ? `intensity MAE ${Math.round(sm.intensity_mae_kt)} kt` : undefined} />
                <div className="neu-inset-sm col-span-2 px-3 py-2 text-[11px] text-muted">
                  <b className="text-ink">Other tracks in these scenes:</b> {sm.confirmed_real} confirmed real systems, <b className={sm.confirmed_false ? "text-sev-orange" : "text-sev-green"}>{sm.confirmed_false} confirmed false alarms</b>
                  <span className="text-faint"> (confirmed = persisted 15 h; {sm.tracks_false - sm.confirmed_false} short-lived false tracks shown faded)</span>
                  {sm.wv_missing && <div className="mt-1 text-sev-orange">Water-vapour channel missing in GridSat for this period — detector uses a WV estimate from IR; intensity runs in IR-only mode.</div>}
                </div>
              </div>
            ) : <Loading h={140} />}
          </Panel>

          <Panel title="Selected system" className="shrink-0"
            right={point ? <ClassPill code={point.imd_class as ClassCode} size="md" /> : null}>
            {point ? (
              <>
                <div className="mb-2 text-[11px] text-muted"><b className="capitalize text-ink">{point.cls}</b> · {point.lat.toFixed(1)}°N {point.lon.toFixed(1)}°E · confidence {Math.round(point.score * 100)}%</div>
                <div className="grid grid-cols-2 gap-2">
                  <Mini label="Image wind (IMD)" value={Math.round(point.wind_imd_kt)} unit="kt" sub={point.truth ? `best track ${Math.round(point.truth.wind_imd_kt)} kt` : "no best-track fix"} />
                  <Mini label="Dvorak T" value={`T${point.t_number.toFixed(1)}`} sub={`${Math.round(point.dvorak_wind_kt)} kt`} />
                  <Mini label="Next 24 h (ConvLSTM)" value={point.seq ? `${point.seq.dv24_kt >= 0 ? "+" : ""}${Math.round(point.seq.dv24_kt)}` : "—"} unit={point.seq ? "kt" : undefined} sub={point.seq ? "from last 21 h of images" : "needs 8 frames of history"} />
                  <Mini label="RI risk · experimental" value={point.seq ? `${Math.round(point.seq.ri_probability * 100)}%` : "—"} sub={point.truth ? `centre error ${Math.round(point.truth.error_km)} km` : undefined} />
                </div>
              </>
            ) : <p className="text-[12px] text-muted">Click a detection on the map.</p>}
          </Panel>

          <Panel title="Intensity from images vs best track" className="min-h-[200px] flex-1"
            right={<span className="flex gap-2 text-[10px] text-muted"><span className="text-accent">━ images</span><span>━ best track</span><span className="text-sev-orange">┅ RI %</span></span>}>
            <div className="h-full">
              {p && series.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={series} margin={{ top: 6, right: 4, left: -24, bottom: 0 }}>
                    <CartesianGrid stroke={p["--sd"]} strokeOpacity={0.5} vertical={false} />
                    <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} ticks={dayTicks(series.map((r) => r.t as number))} stroke={p["--faint"]} fontSize={10} tickLine={false}
                      tickFormatter={(t) => new Date(t).toLocaleString("en-GB", { day: "2-digit", month: "short", timeZone: "UTC" })} />
                    <YAxis yAxisId="w" stroke={p["--faint"]} fontSize={10} tickLine={false} />
                    <YAxis yAxisId="p" orientation="right" domain={[0, 100]} unit="%" stroke={p["--faint"]} fontSize={10} tickLine={false} width={34} />
                    <Tooltip contentStyle={{ background: p["--bg"], border: "none", borderRadius: 14, fontSize: 12 }}
                      labelFormatter={(t) => new Date(t as number).toUTCString().slice(5, 22)} />
                    <Line yAxisId="w" dataKey="best" name="best track kt" stroke={p["--muted"]} strokeWidth={2.5} dot={false} connectNulls isAnimationActive={false} />
                    <Line yAxisId="w" dataKey="image" name="from images kt" stroke={p["--accent"]} strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
                    <Line yAxisId="p" dataKey="ri" name="RI %" stroke={p["--orange"]} strokeWidth={1.5} strokeDasharray="4 3" dot={false} connectNulls isAnimationActive={false} />
                    {storm?.times[pos] && <ReferenceLine yAxisId="w" x={Date.parse(storm.times[pos] + "Z")} stroke={p["--accent"]} />}
                  </ComposedChart>
                </ResponsiveContainer>
              ) : <Loading h={160} />}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
