"use client";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ChevronLeft, ChevronRight, Eye, EyeOff, Pause, Play } from "lucide-react";
import { get, fmtTime, type Analysis, type ImpactData, type RiTimeline, type StormSummary, type Track } from "@/lib/api";
import { ErrorBox, Loading } from "@/components/ui";
import { IntensityChart, RiChart } from "@/components/Charts";
import { AlertsMini, ForecastMini, ImpactMini } from "@/components/RightStack";
import { CurrentMini, GenesisMini, MapLegend, Panel, RiMini, StormPicker, Tabs } from "@/components/Console";

const StormMap = dynamic(() => import("@/components/StormMap"), { ssr: false, loading: () => <Loading h={400} /> });
const FEATURED: [string, number][] = [["BIPARJOY", 2023], ["MOCHA", 2023], ["MICHAUNG", 2023], ["AMPHAN", 2020], ["TAUKTAE", 2021], ["FANI", 2019]];
type ChartTab = "intensity" | "ri";

export default function Operations() {
  const [storms, setStorms] = useState<StormSummary[]>([]);
  const [sid, setSid] = useState<string | null>(null);
  const [track, setTrack] = useState<Track | null>(null);
  const [timeline, setTimeline] = useState<RiTimeline | null>(null);
  const [idx, setIdx] = useState(0);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [impact, setImpact] = useState<ImpactData | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [showActual, setShowActual] = useState(true);
  const [chartTab, setChartTab] = useState<ChartTab>("intensity");
  const cache = useRef(new Map<string, Analysis>());
  // deep link: /?storm=BIPARJOY&season=2023&time=2023-06-13T00:00 (used for demos and the video)
  const linkTime = useRef<string | null>(null);
  const impactCache = useRef(new Map<string, ImpactData>());

  useEffect(() => {
    get<StormSummary[]>("/api/storms").then((s) => {
      setStorms(s);
      const bip = s.find((x) => x.name === "BIPARJOY" && x.season === 2023);
      const q = new URLSearchParams(window.location.search);
      const want = q.get("storm")?.toUpperCase();
      const linked = want ? s.find((x) => x.name === want && (!q.get("season") || String(x.season) === q.get("season"))) : undefined;
      linkTime.current = linked ? q.get("time") : null;
      setSid((linked ?? bip ?? s[0])?.sid ?? null);
    }).catch((e) => setErr(String(e.message ?? e)));
  }, []);

  // load a storm: track + RI timeline; default time = first moment the RI warning would have fired
  useEffect(() => {
    if (!sid) return;
    Promise.all([get<Track>(`/api/storms/${sid}/track`), get<RiTimeline>(`/api/storms/${sid}/ri-timeline`)]).then(([t, tl]) => {
      setTrack(t); setTimeline(tl);
      const ok = (i: number) => t.points[i]?.forecastable;
      let start = tl.points.findIndex((p, i) => ok(i) && p.ri_probability >= tl.threshold);
      if (start < 0) {
        let bestP = -1;
        tl.points.forEach((p, i) => { if (ok(i) && p.ri_probability > bestP) { bestP = p.ri_probability; start = i; } });
      }
      if (linkTime.current) {
        const want = Date.parse(linkTime.current.endsWith("Z") ? linkTime.current : linkTime.current + "Z");
        const near = t.points.reduce((b, p, i) => (Math.abs(Date.parse(p.time + "Z") - want) < Math.abs(Date.parse(t.points[b].time + "Z") - want) ? i : b), 0);
        start = near;
        linkTime.current = null;
      }
      setIdx(Math.max(0, start));
    }).catch((e) => setErr(String(e.message ?? e)));
  }, [sid]);

  const points = useMemo(() => track?.points ?? [], [track]);
  const time = points[idx]?.time;

  useEffect(() => {
    if (!sid || !time) return;
    const key = `${sid}|${time}`;
    const hit = cache.current.get(key);
    if (hit) { setAnalysis(hit); return; }
    setLoading(true);
    const h = setTimeout(() => {
      get<Analysis>(`/api/storms/${sid}/analysis?time=${encodeURIComponent(time)}`)
        .then((a) => { cache.current.set(key, a); setAnalysis(a); setErr(null); })
        .catch((e) => setErr(String(e.message ?? e)))
        .finally(() => setLoading(false));
    }, 120);
    return () => clearTimeout(h);
  }, [sid, time]);

  // impact (landfall ensemble, exposure, sea areas) is slower, so it loads after the analysis
  useEffect(() => {
    if (!sid || !time) return;
    const key = `${sid}|${time}`;
    let live = true;
    const hit = impactCache.current.get(key);
    const h = setTimeout(() => {
      if (hit) { setImpact(hit); return; }
      get<ImpactData>(`/api/storms/${sid}/impact?time=${encodeURIComponent(time)}`)
        .then((d) => { impactCache.current.set(key, d); if (live) setImpact(d); }).catch(() => {});
    }, hit ? 0 : 400);
    return () => { live = false; clearTimeout(h); };
  }, [sid, time]);

  useEffect(() => {
    if (!playing) return;
    const h = setInterval(() => setIdx((i) => { if (i >= points.length - 1) { setPlaying(false); return i; } return i + 1; }), 1500);
    return () => clearInterval(h);
  }, [playing, points.length]);

  const selectStorm = (s: string) => { setPlaying(false); setAnalysis(null); setSid(s); };
  const featured = FEATURED.map(([n, y]) => storms.find((s) => s.name === n && s.season === y)).filter(Boolean) as StormSummary[];
  const liveImpact = impact?.time === analysis?.time ? impact : null;

  if (err && !storms.length) return <ErrorBox msg={err} />;

  return (
    <div className="console-h flex flex-col gap-4">
      {/* control bar */}
      <section className="neu relative z-20 flex shrink-0 flex-wrap items-center gap-3 px-4 py-2.5">
        <StormPicker storms={storms} featured={featured} sid={sid} onSelect={selectStorm} />
        <div className="flex items-center gap-2">
          <button className="neu-btn grid h-10 w-10 place-items-center" onClick={() => setIdx((i) => Math.max(0, i - 1))} aria-label="Previous"><ChevronLeft size={16} /></button>
          <button className="neu-btn grid h-10 w-10 place-items-center text-accent" data-active={playing} aria-label="Play"
            onClick={() => { if (idx >= points.length - 1) setIdx(0); setPlaying((p) => !p); }}>{playing ? <Pause size={16} /> : <Play size={16} />}</button>
          <button className="neu-btn grid h-10 w-10 place-items-center" onClick={() => setIdx((i) => Math.min(points.length - 1, i + 1))} aria-label="Next"><ChevronRight size={16} /></button>
        </div>
        <div className="min-w-[180px] flex-1 px-2">
          <input type="range" className="neu-range" min={0} max={Math.max(0, points.length - 1)} value={idx} onChange={(e) => { setPlaying(false); setIdx(+e.target.value); }} />
        </div>
        <div className="neu-inset-sm px-4 py-1.5 text-center">
          <div className="text-[13px] font-bold tabular text-ink">{time ? fmtTime(time) : "—"}</div>
          <div className="text-[10px] text-faint">uses data up to this time only</div>
        </div>
        <button className="neu-btn flex h-10 items-center gap-2 px-3.5 text-xs font-semibold" data-active={showActual} onClick={() => setShowActual((s) => !s)}>
          {showActual ? <Eye size={14} /> : <EyeOff size={14} />} What actually happened
        </button>
      </section>

      <div className="grid min-h-0 flex-1 gap-4 xl:grid-cols-[minmax(260px,300px)_minmax(0,1fr)_minmax(340px,420px)]">
        {/* left: state + warnings */}
        <div className="flex min-h-0 flex-col gap-4">
          {analysis ? <><CurrentMini a={analysis} /><RiMini a={analysis} /><GenesisMini a={analysis} /></> : <Loading h={600} />}
        </div>

        {/* centre: map + chart strip */}
        <div className="flex min-h-0 flex-col gap-4">
          <section className="neu flex min-h-[420px] flex-1 flex-col p-2">
            <div className="neu-inset relative flex-1">
              <StormMap points={points} idx={idx} analysis={analysis} impact={liveImpact} showActual={showActual} />
              <MapLegend items={[
                [<span key="o" className="h-[3px] w-4 rounded bg-[#6a778b]" />, "Observed"],
                [<span key="f" className="h-[3px] w-4 rounded bg-accent" />, "AI forecast"],
                [<span key="c" className="h-2.5 w-4 rounded-sm bg-accent/25 outline-1 outline-dashed outline-accent" />, "67% cone"],
                ...(showActual ? [[<span key="a" className="h-0 w-4 border-t-2 border-dashed border-[#8a94a6]" />, "Actual track"] as [ReactNode, string]] : []),
                [<span key="d" className="h-2 w-2 rounded-full bg-sev-red" />, "District alert"],
                [<span key="l" className="h-2 w-2 rounded-full bg-[#9b5de5]" />, "Landfall members"],
                [<span key="s" className="h-2.5 w-3.5 rounded-sm border border-dashed border-sev-orange" />, "Sea area"],
              ]} />
              {loading && <div className="neu-sm shimmer absolute right-14 top-3 z-10 px-3 py-1 text-[11px] font-semibold text-muted">updating…</div>}
            </div>
          </section>
          <Panel className="h-[210px] shrink-0"
            title={chartTab === "intensity" ? "Intensity — observed vs forecast (IMD kt)" : "RI probability through the storm's life"}
            right={<Tabs<ChartTab> value={chartTab} onChange={setChartTab} tabs={[{ value: "intensity", label: "Intensity" }, { value: "ri", label: "RI timeline" }]} />}>
            <div className="h-full">
              {chartTab === "intensity" ? <IntensityChart points={points} analysis={analysis} height="100%" /> : <RiChart timeline={timeline} time={time} height="100%" />}
            </div>
          </Panel>
        </div>

        {/* right: forecast, impact and alerts all visible together */}
        <div className="flex min-h-0 flex-col gap-4">
          {analysis ? (
            <>
              <ForecastMini a={analysis} />
              <ImpactMini impact={liveImpact} />
              <AlertsMini key={`${analysis.sid}|${analysis.time}`} a={analysis} impact={liveImpact} />
            </>
          ) : <Loading h={600} />}
        </div>
      </div>
    </div>
  );
}
