"use client";
import { useEffect, useState } from "react";
import { AlertTriangle, Check, Copy, Download, FileText, Sparkles, X } from "lucide-react";
import {
  CLASS_NAMES, FEATURE_LABELS, SEV_VAR, STATIC_MODE, get, getText, kt2kmh,
  type Analysis, type Driver, type Severity, type Sms,
} from "@/lib/api";
import { Card, ClassPill, Gauge, Loading, Segmented, SevDot, Stat } from "./ui";

const compass = (deg: number | null) =>
  deg == null ? "—" : ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"][Math.round(deg / 22.5) % 16];

export function CurrentCard({ a }: { a: Analysis }) {
  const c = a.current;
  return (
    <Card title="Current state" subtitle="Best-track position and IMD classification at the selected time" audience="RSMC forecaster">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <div className="text-2xl font-extrabold text-ink">{a.name === "UNNAMED" ? "Unnamed system" : a.name[0] + a.name.slice(1).toLowerCase()}</div>
          <div className="text-xs text-muted">{CLASS_NAMES[c.imd_class.code]}</div>
        </div>
        <ClassPill code={c.imd_class.code} size="lg" />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <Stat label="Max wind (IMD 3-min)" value={Math.round(c.wind_imd_kt)} unit="kt" sub={`${Math.round(kt2kmh(c.wind_imd_kt))} km/h`} />
        <Stat label="Central pressure" value={Math.round(c.pressure_hpa)} unit="hPa" />
        <Stat label="Movement" value={c.motion_kmh != null ? Math.round(c.motion_kmh) : "—"} unit="km/h" sub={`towards ${compass(c.heading_deg)}`} />
        <Stat label="Distance to land" value={c.dist2land_km != null ? Math.round(c.dist2land_km) : "—"} unit="km"
          sub={c.trend_24h_kt != null ? `${c.trend_24h_kt >= 0 ? "+" : ""}${Math.round(c.trend_24h_kt)} kt in last 24 h` : undefined} />
      </div>
      <div className="mt-3 text-[11px] text-faint tabular">{c.lat.toFixed(1)}°N {c.lon.toFixed(1)}°E</div>
    </Card>
  );
}

export function Drivers({ drivers }: { drivers: Driver[] }) {
  const max = Math.max(...drivers.map((d) => Math.abs(d.impact)), 0.01);
  return (
    <div className="space-y-1.5">
      {drivers.map((d, i) => (
        <div key={i} className="flex items-center gap-2 text-[11px]">
          <span className="w-32 truncate text-muted" title={d.feature}>{FEATURE_LABELS[d.feature] ?? d.feature}</span>
          <div className="neu-inset-sm relative h-2.5 flex-1 overflow-hidden rounded-full">
            <div className="absolute inset-y-0 rounded-full" style={{
              width: `${(Math.abs(d.impact) / max) * 50}%`, left: d.impact >= 0 ? "50%" : undefined, right: d.impact < 0 ? "50%" : undefined,
              background: d.impact >= 0 ? "var(--red)" : "var(--green)", opacity: 0.8 }} />
          </div>
          <span className="w-10 text-right tabular text-faint">{d.value == null ? "—" : Math.abs(d.value) >= 10 ? Math.round(d.value) : d.value.toFixed(1)}</span>
        </div>
      ))}
      <div className="flex justify-between pt-1 text-[10px] text-faint"><span>← lowers risk</span><span>raises risk →</span></div>
    </div>
  );
}

export function RiCard({ a }: { a: Analysis }) {
  const r = a.ri;
  const color = r.flag ? "var(--red)" : r.probability > r.climatology ? "var(--orange)" : "var(--green)";
  return (
    <Card title="Rapid-intensification early warning" subtitle={`P(${r.definition})`} audience="RSMC / SDMA"
      right={r.flag ? <span className="neu-sm flex items-center gap-1 px-2.5 py-1 text-[11px] font-bold text-sev-red"><AlertTriangle size={13} /> Elevated</span> : null}>
      <div className="grid grid-cols-[auto_1fr] items-center gap-5">
        <Gauge value={r.probability} threshold={r.threshold} label="RI risk" color={color} />
        <div>
          <div className="text-3xl font-extrabold tabular" style={{ color }}>{r.ratio_to_climatology.toFixed(1)}×</div>
          <div className="text-xs text-muted">normal risk (climatology {Math.round(r.climatology * 100)}%)</div>
          <div className="mt-2 text-[11px] text-faint">Alert threshold {Math.round(r.threshold * 100)}% · tuned on 2017–19 seasons</div>
          {r.observed != null && (
            <div className="neu-inset-sm mt-3 px-3 py-2 text-[11px]">
              <span className="text-faint">Backtest: </span>
              <span className="font-bold text-ink">{r.observed >= 0 ? "+" : ""}{Math.round(r.observed)} kt</span>
              <span className="text-muted"> in next 24 h — {r.observed >= 30 ? "RI occurred" : "no RI"}</span>
            </div>
          )}
        </div>
      </div>
      <div className="mt-4">
        <div className="label mb-2">Why (SHAP contributions)</div>
        <Drivers drivers={r.drivers} />
      </div>
    </Card>
  );
}

export function GenesisCard({ a }: { a: Analysis }) {
  const g = a.genesis;
  return (
    <Card title="Cyclogenesis outlook" subtitle={`P(${g.definition})`} audience="RSMC forecaster">
      {g.applicable && g.probability != null ? (
        <div className="flex items-center gap-4">
          <div className="neu-inset grid h-20 w-20 shrink-0 place-items-center rounded-full text-xl font-extrabold tabular"
            style={{ color: g.probability >= 0.5 ? "var(--orange)" : "var(--green)" }}>{Math.round(g.probability * 100)}%</div>
          <div className="flex-1"><Drivers drivers={g.drivers.slice(0, 3)} /></div>
        </div>
      ) : (
        <p className="neu-inset-sm px-3 py-3 text-xs text-muted">
          System is already at Cyclonic Storm strength or above — the genesis stage has passed. Scrub back to the depression stage to see the genesis probability.
        </p>
      )}
    </Card>
  );
}

export function ForecastTable({ a, plain = false }: { a: Analysis; plain?: boolean }) {
  const actual = new Map(a.actual.map((x) => [x.lead_h, x]));
  const td = plain ? "px-1.5 py-2" : "p-2";
  return (
    <Card title="Forecast & backtest" subtitle={a.forecast.method} audience="RSMC forecaster" plain={plain}>
      <div className="neu-inset overflow-x-auto p-2">
        <table className={`w-full whitespace-nowrap tabular ${plain ? "text-[11px]" : "text-xs"}`}>
          <thead>
            <tr className="text-left text-faint">
              <th className={td}>Lead</th><th className={td}>{plain ? "Lat, Lon" : "Position"}</th><th className={td}>Wind</th><th className={td}>Class</th>
              <th className={td}>Cone ±</th><th className={td}>Actual error</th>
            </tr>
          </thead>
          <tbody>
            {a.forecast.leads.map((l) => {
              const x = actual.get(l.lead_h);
              return (
                <tr key={l.lead_h} className="border-t border-[color:var(--sd)]/40">
                  <td className={`${td} font-bold text-ink`}>+{l.lead_h}{plain ? "h" : " h"}</td>
                  <td className={`${td} text-muted`}>{plain ? `${l.lat.toFixed(1)}, ${l.lon.toFixed(1)}` : `${l.lat.toFixed(1)}°N ${l.lon.toFixed(1)}°E`}</td>
                  <td className={`${td} text-ink`}>{Math.round(l.wind_imd_kt)} kt</td>
                  <td className={td}><ClassPill code={l.imd_class} size="sm" /></td>
                  <td className={`${td} text-muted`}>{Math.round(l.cone_radius_km)} km</td>
                  <td className={td}>
                    {x ? (
                      <span className="inline-flex items-center gap-1">
                        {x.inside_cone ? <Check size={12} className="text-sev-green" /> : <X size={12} className="text-sev-red" />}
                        <span className="text-ink">{Math.round(x.track_error_km)} km</span>
                        {x.wind_error_kt != null && <span className="text-faint">{x.wind_error_kt > 0 ? "+" : "−"}{Math.abs(Math.round(x.wind_error_kt))} kt</span>}
                      </span>
                    ) : <span className="text-faint">—</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[11px] text-faint">
        Winds shown as IMD 3-min equivalents. Cone = 67% calibrated uncertainty (same 2/3 convention as operational cones). Skill falls off quickly beyond 48 h.
      </p>
    </Card>
  );
}

export type AlertTab = "districts" | "cap" | "sms" | "bulletin";
const LANGS = [{ code: "en", label: "English" }, { code: "hi", label: "हिन्दी" }, { code: "gu", label: "ગુજરાતી" }];

export function AlertsPanel({ a, sms, plain = false, initialTab = "districts" }: { a: Analysis; sms: Sms[] | null; plain?: boolean; initialTab?: AlertTab }) {
  const [tab, setTab] = useState<AlertTab>(initialTab);
  const [langs, setLangs] = useState<string[]>(["en", "hi", "gu"]);
  const [xml, setXml] = useState<{ key: string; text: string } | null>(null);
  const [bulletin, setBulletin] = useState<{ source: string; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const q = `time=${encodeURIComponent(a.time)}`;

  // parent remounts this panel (key) when storm/time changes, so state starts fresh
  const capUrl = `/api/storms/${a.sid}/cap.xml?${q}&langs=${langs.join(",")}`;
  useEffect(() => {
    if (tab !== "cap") return;
    let live = true;
    getText(capUrl).then((text) => live && setXml({ key: capUrl, text }))
      .catch(() => live && setXml({ key: capUrl, text: "<!-- failed to load -->" }));
    return () => { live = false; };
  }, [tab, capUrl]);
  const xmlText = xml?.key === capUrl ? xml.text : null;

  const counts = (["Red", "Orange", "Yellow"] as Severity[]).map((c) => [c, a.impacts.filter((d) => d.color === c).length] as const);
  const copy = (s: string) => { navigator.clipboard?.writeText(s); setCopied(true); setTimeout(() => setCopied(false), 1500); };

  return (
    <Card title="Alerts & dissemination" subtitle="District staging → CAP 1.2 payload in the format NDMA's SACHET system ingests (illustrative, status = Exercise)"
      audience="SDMA / district collector" plain={plain}
      right={<Segmented<AlertTab> value={tab} onChange={setTab} options={[{ value: "districts", label: "Districts" }, { value: "cap", label: "CAP-XML" }, { value: "sms", label: "SMS" }, { value: "bulletin", label: "Bulletin" }]} />}>
      <div className="mb-4 grid grid-cols-3 gap-3">
        {counts.map(([c, n]) => (
          <div key={c} className="neu-inset-sm flex items-center gap-3 px-3 py-2.5">
            <SevDot color={SEV_VAR[c]} />
            <div><div className="text-lg font-extrabold tabular text-ink">{n}</div><div className="text-[10px] font-semibold text-faint">{c} districts</div></div>
          </div>
        ))}
      </div>

      {tab === "districts" && (
        a.impacts.length === 0 ? (
          <p className="neu-inset-sm px-4 py-6 text-center text-sm text-muted">No coastal district inside the 72 h impact envelope at this time.</p>
        ) : (
          <div className={`neu-inset overflow-auto p-2 ${plain ? "max-h-[36vh]" : "max-h-[300px]"}`}>
            <table className="w-full text-xs">
              <thead><tr className="text-left text-faint"><th className="p-2">District</th><th className="p-2">Stage</th><th className="p-2">ETA</th><th className="p-2">Wind</th><th className="p-2">Certainty</th></tr></thead>
              <tbody>
                {a.impacts.map((d) => (
                  <tr key={d.name} className="border-t border-[color:var(--sd)]/40">
                    <td className="p-2"><span className="inline-flex items-center gap-2"><SevDot color={SEV_VAR[d.color]} /><span className="font-semibold text-ink">{d.name}</span><span className="text-faint">{d.state}</span></span></td>
                    <td className="p-2 text-muted">{d.stage}</td>
                    <td className="p-2 tabular text-ink">~{d.eta_h} h</td>
                    <td className="p-2 tabular text-muted">{Math.round(d.max_wind_imd_kt)} kt</td>
                    <td className="p-2 text-muted">{d.certainty}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      )}

      {tab === "cap" && (
        <div>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            {LANGS.map((l) => (
              <button key={l.code} data-active={langs.includes(l.code)} className="neu-btn px-3 py-1.5 text-xs font-semibold"
                onClick={() => setLangs((s) => (s.includes(l.code) ? (s.length > 1 ? s.filter((x) => x !== l.code) : s) : [...s, l.code]))}>
                {l.label}
              </button>
            ))}
            <div className="ml-auto flex gap-2">
              <button className="neu-btn flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold" disabled={!xmlText} onClick={() => xmlText && copy(xmlText)}>
                {copied ? <Check size={13} /> : <Copy size={13} />} Copy
              </button>
              <button className="neu-btn flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold" disabled={!xmlText}
                onClick={() => {
                  if (!xmlText) return;
                  const url = URL.createObjectURL(new Blob([xmlText], { type: "application/xml" }));
                  const link = Object.assign(document.createElement("a"), { href: url, download: `cap-${a.sid}-${a.time.slice(0, 13)}.xml` });
                  link.click();
                  URL.revokeObjectURL(url);
                }}>
                <Download size={13} /> Download
              </button>
            </div>
          </div>
          {xmlText ? <pre className={`xml neu-inset overflow-auto p-4 font-mono text-[11px] leading-relaxed text-muted ${plain ? "max-h-[38vh]" : "max-h-[320px]"}`}>{xmlText}</pre> : <Loading h={320} />}
        </div>
      )}

      {tab === "sms" && (
        sms ? (
          <div className="space-y-3">
            {sms.map((m) => (
              <div key={m.lang} className="neu-inset-sm px-4 py-3">
                <div className="mb-1.5 flex items-center justify-between text-[10px] font-semibold text-faint">
                  <span>{LANGS.find((l) => l.code === m.lang)?.label}</span>
                  <span className={m.segments > 1 ? "text-sev-orange" : "text-sev-green"}>{m.encoding} · {m.chars} chars · {m.segments} SMS</span>
                </div>
                <p className="text-sm text-ink">{m.text}</p>
              </div>
            ))}
            <p className="text-[11px] text-faint">Indian-language SMS use UCS-2 encoding (70 characters per segment), so they are drafted short to limit segments on congested networks.</p>
          </div>
        ) : <Loading h={200} />
      )}

      {tab === "bulletin" && (
        <div>
          {!bulletin && (
            <button disabled={busy} className="neu-btn flex items-center gap-2 px-4 py-2.5 text-sm font-semibold"
              onClick={() => { setBusy(true); get<{ source: string; text: string }>(`/api/storms/${a.sid}/bulletin?${q}`).then(setBulletin).finally(() => setBusy(false)); }}>
              {busy ? <Sparkles size={15} className="shimmer" /> : <FileText size={15} />} {busy ? "Writing bulletin…" : "Generate text bulletin"}
            </button>
          )}
          {bulletin && (
            <>
              <div className="mb-2 text-[11px] text-faint">Source: {bulletin.source.startsWith("gemini") ? `${bulletin.source} (constrained to model numbers)` : STATIC_MODE ? "deterministic template (live Gemini wording runs in the full version)" : "deterministic template"}</div>
              <pre className="neu-inset max-h-[320px] overflow-auto whitespace-pre-wrap p-4 font-mono text-[11px] leading-relaxed text-muted">{bulletin.text}</pre>
            </>
          )}
        </div>
      )}
    </Card>
  );
}
