"use client";
import {
  Area, Bar, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { usePalette } from "@/lib/theme";
import type { Analysis, RiTimeline, TrackPoint } from "@/lib/api";

const ms = (iso: string) => Date.parse(iso.endsWith("Z") ? iso : iso + "Z");
const tick = (t: number) => new Date(t).toLocaleString("en-GB", { day: "2-digit", month: "short", timeZone: "UTC" });
/** Unique 00 UTC ticks across the data range (every 2nd day for long storms). Explicit ticks avoid
 *  recharts' auto time ticks, which can repeat a value and trigger duplicate React keys. */
export function dayTicks(ts: number[]): number[] {
  const f = ts.filter(Number.isFinite);
  if (!f.length) return [];
  const DAY = 86400e3;
  const lo = Math.ceil(Math.min(...f) / DAY) * DAY, hi = Math.max(...f);
  const step = hi - lo > 12 * DAY ? 2 * DAY : DAY;
  const out: number[] = [];
  for (let t = lo; t <= hi; t += step) out.push(t);
  return out;
}

const tipTime = (t: number) => new Date(t).toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" }) + " UTC";

function tooltipStyle(p: ReturnType<typeof usePalette>) {
  return {
    contentStyle: { background: p?.["--bg"], border: "none", borderRadius: 14, boxShadow: `6px 6px 14px ${p?.["--sd"]}`, fontSize: 12, color: p?.["--text"] },
    labelStyle: { color: p?.["--muted"], fontWeight: 600 },
  };
}

export function IntensityChart({ points, analysis, height = 230 }: { points: TrackPoint[]; analysis: Analysis | null; height?: number | `${number}%` }) {
  const p = usePalette();
  if (!p) return <div style={{ height }} />;
  const t0 = analysis ? ms(analysis.time) : null;
  const rows: Record<string, number | null | number[]>[] = points.map((pt) => ({
    t: ms(pt.time), observed: pt.wind_imd_kt != null ? Math.round(pt.wind_imd_kt) : null,
  }));
  if (analysis && t0) {
    rows.push({ t: t0, forecast: Math.round(analysis.current.wind_imd_kt), band: [analysis.current.wind_imd_kt, analysis.current.wind_imd_kt] });
    analysis.forecast.leads.forEach((l) => {
      const sd = l.wind_sd_kt * 0.93;
      rows.push({ t: t0 + l.lead_h * 3600e3, forecast: Math.round(l.wind_imd_kt), band: [Math.max(0, l.wind_imd_kt - sd), l.wind_imd_kt + sd] });
    });
  }
  const byT = new Map<number, Record<string, number | null | number[]>>();
  rows.forEach((r) => byT.set(r.t as number, { ...(byT.get(r.t as number) ?? {}), ...r }));
  const merged = [...byT.values()].sort((a, b) => (a.t as number) - (b.t as number));
  const cls = [28, 34, 48, 64, 90, 120];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={merged} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
        <CartesianGrid stroke={p["--sd"]} strokeOpacity={0.5} vertical={false} />
        <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} ticks={dayTicks(merged.map((r) => r.t as number))} tickFormatter={tick} stroke={p["--faint"]} fontSize={11} tickLine={false} />
        <YAxis stroke={p["--faint"]} fontSize={11} tickLine={false} domain={[0, (max: number) => Math.max(70, Math.ceil(max / 10) * 10 + 10)]} />
        {cls.map((c) => <ReferenceLine key={c} y={c} stroke={p["--faint"]} strokeOpacity={0.35} strokeDasharray="2 4" />)}
        <Tooltip labelFormatter={(t) => tipTime(t as number)} formatter={(v, n) => [Array.isArray(v) ? `${Math.round(v[0] as number)}–${Math.round(v[1] as number)} kt` : `${v} kt`, n === "band" ? "forecast spread" : n]} {...tooltipStyle(p)} />
        <Area dataKey="band" stroke="none" fill={p["--accent"]} fillOpacity={0.15} connectNulls isAnimationActive={false} />
        <Line dataKey="observed" name="observed (IMD 3-min)" stroke={p["--muted"]} strokeWidth={2.5} dot={false} connectNulls isAnimationActive={false} />
        <Line dataKey="forecast" name="forecast" stroke={p["--accent"]} strokeWidth={2.5} strokeDasharray="6 4" dot={{ r: 3 }} connectNulls isAnimationActive={false} />
        {t0 && <ReferenceLine x={t0} stroke={p["--accent"]} strokeWidth={1.5} label={{ value: "now", position: "top", fill: p["--accent"], fontSize: 11 }} />}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function RiChart({ timeline, time, height = 200 }: { timeline: RiTimeline | null; time?: string; height?: number | `${number}%` }) {
  const p = usePalette();
  if (!p || !timeline) return <div style={{ height }} />;
  const rows = timeline.points.map((x) => ({
    t: ms(x.time), p: Math.round(x.ri_probability * 1000) / 10,
    ri: x.observed_dw24 != null && x.observed_dw24 >= 30 ? 100 : null,
  }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
        <CartesianGrid stroke={p["--sd"]} strokeOpacity={0.5} vertical={false} />
        <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} ticks={dayTicks(rows.map((r) => r.t))} tickFormatter={tick} stroke={p["--faint"]} fontSize={11} tickLine={false} />
        <YAxis stroke={p["--faint"]} fontSize={11} tickLine={false} domain={[0, (m: number) => Math.min(100, Math.max(40, Math.ceil(m / 10) * 10))]} unit="%" />
        <Tooltip labelFormatter={(t) => tipTime(t as number)} formatter={(v, n) => (n === "ri" ? ["RI occurred in next 24 h", ""] : [`${v}%`, "RI probability"])} {...tooltipStyle(p)} />
        <Bar dataKey="ri" fill={p["--red"]} fillOpacity={0.14} barSize={8} isAnimationActive={false} />
        <Area dataKey="p" stroke={p["--orange"]} strokeWidth={2.5} fill={p["--orange"]} fillOpacity={0.12} isAnimationActive={false} />
        <ReferenceLine y={timeline.threshold * 100} stroke={p["--red"]} strokeDasharray="5 4" label={{ value: "alert threshold", position: "insideTopRight", fill: p["--red"], fontSize: 10 }} />
        <ReferenceLine y={timeline.climatology * 100} stroke={p["--faint"]} strokeDasharray="2 4" label={{ value: "climatology", position: "insideBottomRight", fill: p["--faint"], fontSize: 10 }} />
        {time && <ReferenceLine x={ms(time)} stroke={p["--accent"]} strokeWidth={1.5} />}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
