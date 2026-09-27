"use client";
/** Right-hand column of the Operations console: Forecast, Impact and Alerts visible together. */
import { useEffect, useState, type ReactNode } from "react";
import { Check, FileCode2, FileText, MessageSquare, X } from "lucide-react";
import { SEV_VAR, fmtPeople, type Analysis, type ImpactData, type Severity } from "@/lib/api";
import { AlertsPanel, type AlertTab } from "./OpsPanels";
import { Panel } from "./Console";
import { ClassPill, Loading, SevDot } from "./ui";

const IST = (iso: string) =>
  new Date(iso + (iso.endsWith("Z") ? "" : "Z")).toLocaleString("en-IN", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "Asia/Kolkata" });

export function ForecastMini({ a }: { a: Analysis }) {
  const actual = new Map(a.actual.map((x) => [x.lead_h, x]));
  return (
    <Panel title="Forecast & backtest" className="shrink-0"
      right={<span className="max-w-[55%] truncate text-[10px] text-faint" title={a.forecast.method}>{a.forecast.method.split(" (")[0]}</span>}>
      <table className="w-full whitespace-nowrap text-[11px] tabular">
        <thead>
          <tr className="text-left text-[10px] text-faint">
            <th className="pb-1 font-semibold">Lead</th><th className="pb-1 font-semibold">Wind</th><th className="pb-1 font-semibold">Class</th>
            <th className="pb-1 font-semibold">Cone ±</th><th className="pb-1 font-semibold">Actual error</th>
          </tr>
        </thead>
        <tbody>
          {a.forecast.leads.map((l) => {
            const x = actual.get(l.lead_h);
            return (
              <tr key={l.lead_h} className="border-t border-[color:var(--sd)]/40">
                <td className="py-1 font-bold text-ink">+{l.lead_h}h</td>
                <td className="py-1 text-ink">{Math.round(l.wind_imd_kt)} kt</td>
                <td className="py-1"><ClassPill code={l.imd_class} size="sm" /></td>
                <td className="py-1 text-muted">{Math.round(l.cone_radius_km)} km</td>
                <td className="py-1">
                  {x ? (
                    <span className="inline-flex items-center gap-1">
                      {x.inside_cone ? <Check size={11} className="text-sev-green" /> : <X size={11} className="text-sev-red" />}
                      <span className="text-ink">{Math.round(x.track_error_km)} km</span>
                    </span>
                  ) : <span className="text-faint">—</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Panel>
  );
}

function Cell({ label, value, sub }: { label: string; value: ReactNode; sub?: string }) {
  return (
    <div className="neu-inset-sm min-w-0 px-2.5 py-1.5">
      <div className="text-[9px] font-bold uppercase tracking-wider text-faint">{label}</div>
      <div className="truncate text-[13px] font-bold leading-tight tabular text-ink">{value}</div>
      {sub && <div className="truncate text-[9.5px] text-muted" title={sub}>{sub}</div>}
    </div>
  );
}

export function ImpactMini({ impact }: { impact: ImpactData | null }) {
  if (!impact) return <Panel title="Impact" className="shrink-0"><Loading h={120} /></Panel>;
  const lf = impact.landfall, ex = impact.exposure;
  const likely = lf.status === "possible" && (lf.probability_72h ?? 0) >= 0.05 && lf.point;
  const seas = impact.sea_areas;
  return (
    <Panel title="Landfall, exposure & fishermen" className="shrink-0">
      <div className="grid grid-cols-2 gap-1.5">
        <Cell label="Landfall in 72 h" value={lf.status === "possible" ? `${Math.round((lf.probability_72h ?? 0) * 100)}%` : lf.status === "over_land" ? "over land" : "—"}
          sub={lf.india_probability != null ? `India coast ${Math.round(lf.india_probability * 100)}%` : lf.status === "none" ? "no member reaches land" : undefined} />
        <Cell label="Most likely at" value={likely ? lf.country : "—"}
          sub={likely && lf.nearest_district ? `${lf.nearest_district.name} ${Math.round(lf.nearest_district.distance_km)} km` : undefined} />
        <Cell label="Most likely time (IST)" value={likely && lf.time_median ? IST(lf.time_median) : "—"}
          sub={likely && lf.window ? `80%: ${IST(lf.window[0])} → ${IST(lf.window[1])}` : undefined} />
        <Cell label="People in gale zone" value={fmtPeople(ex.people_in_gale_envelope)}
          sub={`${fmtPeople(ex.by_window["0-24h"] ?? 0)} within 24 h`} />
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {seas.length === 0 ? <span className="text-[10.5px] text-muted">No fishermen warning for any sea area.</span> : (
          <>
            {seas.slice(0, 4).map((s) => (
              <span key={s.area} className="neu-sm inline-flex items-center gap-1.5 px-2 py-1 text-[10px] text-ink" title={`${s.sea_state} · ${s.action}`}>
                <SevDot color={SEV_VAR[s.color]} />{s.area.replace(" Bay of Bengal", " BoB").replace(" Arabian Sea", " AS")}
                <span className="text-faint">{s.from_h === 0 ? "now" : `+${s.from_h}h`}</span>
              </span>
            ))}
            {seas.length > 4 && <span className="self-center text-[10px] text-faint">+{seas.length - 4} more</span>}
          </>
        )}
      </div>
    </Panel>
  );
}

export function AlertsMini({ a, impact }: { a: Analysis; impact: ImpactData | null }) {
  const [open, setOpen] = useState<AlertTab | null>(null);
  const counts = (["Red", "Orange", "Yellow"] as Severity[]).map((c) => [c, a.impacts.filter((d) => d.color === c).length] as const);
  return (
    <Panel title="District alerts" className="min-h-[170px] flex-1" bodyClass="flex flex-col min-h-0"
      right={<div className="flex gap-1.5">
        {counts.map(([c, n]) => (
          <span key={c} className="neu-inset-sm inline-flex items-center gap-1.5 px-2 py-0.5 text-[11px] font-bold tabular text-ink"><SevDot color={SEV_VAR[c]} />{n}</span>
        ))}
      </div>}>
      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        {a.impacts.length === 0 ? (
          <p className="neu-inset-sm px-3 py-3 text-center text-[11px] text-muted">No coastal district inside the 72 h impact envelope at this time.</p>
        ) : (
          <table className="w-full whitespace-nowrap text-[11px]">
            <tbody>
              {a.impacts.map((d) => (
                <tr key={d.name} className="border-t border-[color:var(--sd)]/40 first:border-0">
                  <td className="py-1"><span className="inline-flex items-center gap-1.5"><SevDot color={SEV_VAR[d.color]} /><b className="text-ink">{d.name}</b></span></td>
                  <td className="max-w-[74px] truncate py-1 pl-2 text-faint" title={d.state}>{d.state}</td>
                  <td className="py-1 pl-2 text-muted">{d.stage.replace("Cyclone ", "").replace("Pre-Cyclone ", "")}</td>
                  <td className="py-1 pl-2 text-right tabular text-ink">~{d.eta_h}h</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <div className="mt-2 grid shrink-0 grid-cols-3 gap-2">
        <button className="neu-btn flex items-center justify-center gap-1.5 py-2 text-[11px] font-semibold" onClick={() => setOpen("cap")}><FileCode2 size={13} /> CAP-XML</button>
        <button className="neu-btn flex items-center justify-center gap-1.5 py-2 text-[11px] font-semibold" onClick={() => setOpen("sms")}><MessageSquare size={13} /> SMS</button>
        <button className="neu-btn flex items-center justify-center gap-1.5 py-2 text-[11px] font-semibold" onClick={() => setOpen("bulletin")}><FileText size={13} /> Bulletin</button>
      </div>
      {open && (
        <Modal onClose={() => setOpen(null)}>
          <AlertsPanel key={`${a.sid}|${a.time}|${open}`} a={a} sms={impact?.sms ?? null} initialTab={open} />
        </Modal>
      )}
    </Panel>
  );
}

function Modal({ children, onClose }: { children: ReactNode; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/25 p-4 backdrop-blur-[2px]" onClick={onClose} role="dialog" aria-modal="true">
      <div className="relative w-full max-w-4xl pt-12" onClick={(e) => e.stopPropagation()}>
        <button onClick={onClose} className="neu-btn absolute right-0 top-0 z-10 flex h-9 items-center gap-1.5 px-3 text-xs font-semibold" aria-label="Close">
          <X size={15} /> Close
        </button>
        <div className="max-h-[80vh] overflow-y-auto rounded-[26px]">{children}</div>
      </div>
    </div>
  );
}
