"use client";
import { Anchor, MapPin, Users } from "lucide-react";
import { SEV_VAR, fmtPeople, fmtTime, type ImpactData } from "@/lib/api";
import { Card, Loading, SevDot, Stat } from "./ui";

const IST = (iso: string) =>
  new Date(iso + (iso.endsWith("Z") ? "" : "Z")).toLocaleString("en-IN", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "Asia/Kolkata" }) + " IST";

export function LandfallCard({ impact, plain = false }: { impact: ImpactData | null; plain?: boolean }) {
  if (!impact) return <Loading h={330} />;
  const lf = impact.landfall, ex = impact.exposure;
  return (
    <Card title="Landfall & exposure" subtitle="From 50 calibrated forecast members · population: WorldPop 2020" audience="SDMA / district collector" plain={plain}>
      {lf.status === "possible" && (lf.probability_72h ?? 0) < 0.05 ? (
        <p className="neu-inset-sm px-4 py-3 text-xs text-muted">
          <MapPin size={13} className="mr-1 inline" />Landfall unlikely within 72 h — only {Math.round((lf.probability_72h ?? 0) * 100)}% of forecast members reach land ({lf.country}).
        </p>
      ) : lf.status === "possible" && lf.point ? (
        <>
          <div className="grid grid-cols-2 gap-3">
            <Stat label="Landfall within 72 h" value={`${Math.round((lf.probability_72h ?? 0) * 100)}%`}
              sub={lf.india_probability != null ? `on the Indian coast: ${Math.round(lf.india_probability * 100)}%` : undefined} />
            <Stat label="Most likely at" value={<span className="text-base">{lf.country}</span>}
              sub={lf.nearest_district ? `nearest Indian district: ${lf.nearest_district.name} (${Math.round(lf.nearest_district.distance_km)} km)` : undefined} />
            <Stat label="Most likely time" value={<span className="text-base">{lf.time_median ? IST(lf.time_median) : "—"}</span>}
              sub={lf.window ? `80% window: ${IST(lf.window[0])} → ${IST(lf.window[1])}` : undefined} />
            <Stat label="Wind at landfall" value={lf.wind_imd_kt ? Math.round(lf.wind_imd_kt[1]) : "—"} unit="kt"
              sub={lf.wind_imd_kt ? `range ${Math.round(lf.wind_imd_kt[0])}–${Math.round(lf.wind_imd_kt[2])} kt` : undefined} />
          </div>
          <p className="mt-2 text-[11px] text-faint">Landfall points of the members are shown as small dots on the map; 80% of them fall within {Math.round(lf.spread_km_80pct ?? 0)} km of the median point.</p>
        </>
      ) : (
        <p className="neu-inset-sm px-4 py-3 text-xs text-muted">
          <MapPin size={13} className="mr-1 inline" />
          {lf.status === "over_land" ? "The centre is already over land." : lf.status === "none" ? "No forecast member reaches land within 72 h." : lf.message}
        </p>
      )}
      <div className="neu-inset-sm mt-4 px-4 py-3">
        <div className="flex items-center gap-3">
          <Users size={18} className="text-accent" />
          <div>
            <div className="text-xl font-extrabold tabular text-ink">{fmtPeople(ex.people_in_gale_envelope)}</div>
            <div className="text-[11px] text-muted">people inside the damaging-wind envelope (0–72 h)</div>
          </div>
        </div>
        <div className="mt-3 grid grid-cols-3 gap-2 text-center text-[11px]">
          {Object.entries(ex.by_window).map(([w, n]) => (
            <div key={w} className="neu-sm px-2 py-1.5"><div className="font-bold tabular text-ink">{fmtPeople(n)}</div><div className="text-faint">within {w.replace("0-", "")}</div></div>
          ))}
        </div>
        {Object.keys(ex.by_country).length > 0 && (
          <div className="mt-2 text-[11px] text-muted">{Object.entries(ex.by_country).map(([c, n]) => `${c} ${fmtPeople(n)}`).join(" · ")}</div>
        )}
      </div>
    </Card>
  );
}

export function SeaAreasCard({ impact, plain = false }: { impact: ImpactData | null; plain?: boolean }) {
  if (!impact) return <Loading h={330} />;
  return (
    <Card title="Fishermen warnings by sea area" subtitle="Sea areas the storm's damaging winds may reach in the next 72 h (illustrative boxes)" audience="Fisheries dept / coastal communities" plain={plain}>
      {impact.sea_areas.length === 0 ? (
        <p className="neu-inset-sm px-4 py-6 text-center text-sm text-muted"><Anchor size={14} className="mr-1 inline" />No sea area affected in the next 72 h.</p>
      ) : (
        <div className={`neu-inset overflow-auto p-2 ${plain ? "" : "max-h-[330px]"}`}>
          <table className="w-full text-xs">
            <thead><tr className="text-left text-faint"><th className="p-2">Sea area</th><th className="p-2">From</th><th className="p-2">Sea state</th><th className="p-2">Advice</th></tr></thead>
            <tbody>
              {impact.sea_areas.map((s) => (
                <tr key={s.area} className="border-t border-[color:var(--sd)]/40 align-top">
                  <td className="p-2"><span className="inline-flex items-center gap-2 font-semibold text-ink"><SevDot color={SEV_VAR[s.color]} />{s.area}</span></td>
                  <td className="whitespace-nowrap p-2 tabular text-muted">{s.from_h === 0 ? "now" : `+${s.from_h} h`}</td>
                  <td className="p-2 text-muted">{s.sea_state}<div className="text-faint">winds to {Math.round(s.max_wind_kt)} kt</div></td>
                  <td className="p-2 text-muted">{s.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="mt-2 text-[11px] text-faint">Valid from {fmtTime(impact.time)}. Sea-state wording follows common marine-bulletin categories; boxes approximate the named sea areas.</p>
    </Card>
  );
}
