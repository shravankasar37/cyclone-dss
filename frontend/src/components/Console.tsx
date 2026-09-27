"use client";
/** Compact building blocks for the single-screen console layouts. */
import { useState, type ReactNode } from "react";
import { AlertTriangle, ChevronDown, Search } from "lucide-react";
import { CLASS_NAMES, kt2kmh, type Analysis, type StormSummary } from "@/lib/api";
import { Drivers } from "./OpsPanels";
import { ClassPill, Gauge, Segmented } from "./ui";

const titleCase = (s: string) => (s === "UNNAMED" ? "Unnamed" : s[0] + s.slice(1).toLowerCase());
const compass = (deg: number | null) =>
  deg == null ? "—" : ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"][Math.round(deg / 22.5) % 16];

/** Raised panel that fills its grid cell; body scrolls internally only if the window is very short. */
export function Panel({ title, right, children, className = "", bodyClass = "" }: {
  title?: string; right?: ReactNode; children: ReactNode; className?: string; bodyClass?: string;
}) {
  return (
    <section className={`neu flex min-h-0 flex-col p-4 ${className}`}>
      {(title || right) && (
        <header className="mb-3 flex shrink-0 items-center justify-between gap-2">
          {title && <h2 className="text-[13px] font-bold text-ink">{title}</h2>}
          {right}
        </header>
      )}
      <div className={`min-h-0 flex-1 ${bodyClass}`}>{children}</div>
    </section>
  );
}

export function StormPicker({ storms, featured, sid, onSelect }: {
  storms: StormSummary[]; featured: StormSummary[]; sid: string | null; onSelect: (sid: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const cur = storms.find((s) => s.sid === sid);
  const list = storms.filter((s) => `${s.name} ${s.season}`.toLowerCase().includes(q.toLowerCase())).slice(0, 40);
  const pick = (s: string) => { onSelect(s); setOpen(false); setQ(""); };
  return (
    <div className="relative">
      <button className="neu-btn flex h-11 items-center gap-2 px-4 text-sm font-bold text-ink" data-active={open} onClick={() => setOpen((o) => !o)}>
        {cur ? <>{titleCase(cur.name)} <span className="font-medium text-faint">{cur.season}</span></> : "Choose storm"}
        <ChevronDown size={15} />
      </button>
      {open && (
        <div className="neu absolute left-0 z-30 mt-3 w-80 p-3" onMouseLeave={() => setOpen(false)}>
          <div className="label mb-2">Demo storms</div>
          <div className="mb-3 flex flex-wrap gap-2">
            {featured.map((s) => (
              <button key={s.sid} data-active={s.sid === sid} onClick={() => pick(s.sid)} className="neu-btn px-2.5 py-1.5 text-[11px] font-bold">
                {titleCase(s.name)} <span className="font-medium text-faint">{s.season}</span>
              </button>
            ))}
          </div>
          <div className="neu-inset-sm mb-2 flex items-center gap-2 px-3 py-2">
            <Search size={13} className="text-faint" />
            <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder={`Search ${storms.length} storms since ${storms.length ? Math.min(...storms.map((s) => s.season)) : ""}`}
              className="w-full bg-transparent text-xs text-ink outline-none placeholder:text-faint" />
          </div>
          <div className="max-h-60 overflow-auto">
            {list.map((s) => (
              <button key={s.sid} onClick={() => pick(s.sid)} className="flex w-full items-center justify-between rounded-xl px-2.5 py-1.5 text-left text-xs hover:bg-[color:var(--sd)]/20">
                <span className="font-semibold text-ink">{titleCase(s.name)} <span className="text-faint">{s.season} · {s.subbasin === "AS" ? "Arabian Sea" : "Bay of Bengal"}</span></span>
                <ClassPill code={s.peak_class} size="sm" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function Mini({ label, value, unit, sub }: { label: string; value: ReactNode; unit?: string; sub?: ReactNode }) {
  return (
    <div className="neu-inset-sm px-3 py-2">
      <div className="text-[9.5px] font-bold uppercase tracking-wider text-faint">{label}</div>
      <div className="text-base font-bold leading-tight tabular text-ink">{value}{unit && <span className="ml-1 text-[10px] font-semibold text-muted">{unit}</span>}</div>
      {sub && <div className="truncate text-[10px] text-muted">{sub}</div>}
    </div>
  );
}

export function CurrentMini({ a }: { a: Analysis }) {
  const c = a.current;
  return (
    <Panel>
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-lg font-extrabold text-ink">{titleCase(a.name)}</div>
          <div className="truncate text-[11px] text-muted">{CLASS_NAMES[c.imd_class.code]} · {c.lat.toFixed(1)}°N {c.lon.toFixed(1)}°E</div>
        </div>
        <ClassPill code={c.imd_class.code} size="md" />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Mini label={c.wind_estimated ? "Wind (IMD, est.)" : "Wind (IMD)"} value={Math.round(c.wind_imd_kt)} unit="kt" sub={c.wind_estimated ? "interpolated: no report at this fix" : `${Math.round(kt2kmh(c.wind_imd_kt))} km/h`} />
        <Mini label="Pressure" value={Math.round(c.pressure_hpa)} unit="hPa" />
        <Mini label="Moving" value={c.motion_kmh != null ? Math.round(c.motion_kmh) : "—"} unit="km/h" sub={`towards ${compass(c.heading_deg)}`} />
        <Mini label="To land" value={c.dist2land_km != null ? Math.round(c.dist2land_km) : "—"} unit="km"
          sub={c.trend_24h_kt != null ? `${c.trend_24h_kt >= 0 ? "+" : ""}${Math.round(c.trend_24h_kt)} kt / 24 h` : undefined} />
      </div>
    </Panel>
  );
}

export function RiMini({ a }: { a: Analysis }) {
  const r = a.ri;
  const color = r.flag ? "var(--red)" : r.probability > r.climatology ? "var(--orange)" : "var(--green)";
  return (
    <Panel title="Rapid-intensification warning" className="flex-1" bodyClass="overflow-y-auto"
      right={r.flag ? <span className="flex items-center gap-1 text-[11px] font-bold text-sev-red"><AlertTriangle size={12} /> Elevated</span> : null}>
      <div className="flex items-center gap-4">
        <Gauge small value={r.probability} threshold={r.threshold} label="RI risk" color={color} />
        <div className="min-w-0">
          <div className="text-2xl font-extrabold tabular" style={{ color }}>{r.ratio_to_climatology.toFixed(1)}×</div>
          <div className="text-[11px] text-muted">normal risk ({Math.round(r.climatology * 100)}% climatology)</div>
          <div className="text-[10px] text-faint">≥30 kt in 24 h · alert at {Math.round(r.threshold * 100)}%</div>
          {r.observed != null && (
            <div className="mt-1.5 text-[11px]"><span className="text-faint">Actual: </span>
              <b className="tabular text-ink">{r.observed >= 0 ? "+" : ""}{Math.round(r.observed)} kt</b>
              <span className={r.observed >= 30 ? " text-sev-red" : " text-muted"}> {r.observed >= 30 ? "RI occurred" : "no RI"}</span></div>
          )}
        </div>
      </div>
      <div className="mt-3"><div className="label mb-1.5">Why</div><Drivers drivers={r.drivers.slice(0, 4)} /></div>
    </Panel>
  );
}

export function GenesisMini({ a }: { a: Analysis }) {
  const g = a.genesis;
  return (
    <Panel>
      <div className="flex items-center gap-3">
        <div className="neu-inset grid h-14 w-14 shrink-0 place-items-center rounded-full text-base font-extrabold tabular"
          style={{ color: g.probability == null ? "var(--faint)" : g.probability >= 0.5 ? "var(--orange)" : "var(--green)" }}>
          {g.applicable && g.probability != null ? `${Math.round(g.probability * 100)}%` : "—"}
        </div>
        <div className="min-w-0">
          <div className="text-[13px] font-bold text-ink">Cyclogenesis outlook</div>
          <div className="text-[11px] text-muted">{g.applicable ? "chance this depression becomes a Cyclonic Storm within 72 h" : `Not applicable — ${g.reason}`}</div>
        </div>
      </div>
    </Panel>
  );
}

export function MapLegend({ items }: { items: [ReactNode, string][] }) {
  return (
    <div className="neu-sm pointer-events-none absolute bottom-3 left-3 z-10 flex max-w-[70%] flex-wrap gap-x-3 gap-y-1 px-3 py-2 text-[10px] text-muted">
      {items.map(([sw, label]) => <span key={label} className="flex items-center gap-1.5">{sw}{label}</span>)}
    </div>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { value: T; label: string }[]; value: T; onChange: (v: T) => void }) {
  return <Segmented options={tabs} value={value} onChange={onChange} />;
}
