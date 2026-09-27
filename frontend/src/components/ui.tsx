import type { ReactNode } from "react";
import { CLASS_COLORS, CLASS_NAMES, type ClassCode } from "@/lib/api";

export function Card({ title, subtitle, right, children, className = "", audience, plain = false }: {
  title?: string; subtitle?: string; right?: ReactNode; children: ReactNode; className?: string; audience?: string;
  /** plain: no raised surface or title - for panels embedded in a tab of another card */
  plain?: boolean;
}) {
  if (plain) {
    return (
      <div className={className}>
        {(subtitle || right) && (
          <div className="mb-3 flex items-start justify-between gap-3">
            {subtitle ? <p className="text-[11px] text-muted">{subtitle}</p> : <span />}
            {right}
          </div>
        )}
        {children}
      </div>
    );
  }
  return (
    <section className={`neu p-5 ${className}`}>
      {(title || right) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-[15px] font-bold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          <div className="flex items-center gap-2">
            {audience && <span className="neu-inset-sm px-2.5 py-1 text-[10px] font-semibold text-faint">For: {audience}</span>}
            {right}
          </div>
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, unit, sub }: { label: string; value: ReactNode; unit?: string; sub?: ReactNode }) {
  return (
    <div className="neu-inset-sm px-3.5 py-3">
      <div className="label">{label}</div>
      <div className="mt-1 text-xl font-bold tabular text-ink">
        {value}
        {unit && <span className="ml-1 text-xs font-semibold text-muted">{unit}</span>}
      </div>
      {sub && <div className="mt-0.5 text-[11px] text-muted">{sub}</div>}
    </div>
  );
}

export function ClassPill({ code, size = "md" }: { code: ClassCode; size?: "sm" | "md" | "lg" }) {
  const pad = size === "lg" ? "px-4 py-2 text-base" : size === "sm" ? "px-2 py-0.5 text-[11px]" : "px-3 py-1 text-xs";
  return (
    <span className={`neu-sm inline-flex items-center gap-2 font-bold ${pad}`} title={CLASS_NAMES[code]}>
      <span className="h-2.5 w-2.5 rounded-full" style={{ background: CLASS_COLORS[code] }} />
      <span style={{ color: CLASS_COLORS[code] }}>{code}</span>
    </span>
  );
}

export function SevDot({ color }: { color: string }) {
  return <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: color }} />;
}

/** Neumorphic circular gauge. value in [0,1]; marker at threshold. */
export function Gauge({ value, threshold, label, color, small = false }: { value: number; threshold?: number; label: string; color: string; small?: boolean }) {
  const r = 52, c = 2 * Math.PI * r, arc = 0.75;
  const v = Math.max(0, Math.min(1, value));
  const tAngle = threshold !== undefined ? 225 + 270 * threshold : null; // arc starts at 225° (clockwise from top)
  return (
    <div className={`neu relative mx-auto grid shrink-0 place-items-center rounded-full ${small ? "h-28 w-28" : "h-40 w-40"}`}>
      <div className={`neu-inset absolute rounded-full ${small ? "inset-3" : "inset-4"}`} />
      <svg viewBox="0 0 140 140" className="absolute inset-0 h-full w-full -rotate-[225deg]">
        <circle cx="70" cy="70" r={r} fill="none" stroke="var(--sd)" strokeOpacity=".35" strokeWidth="9"
          strokeDasharray={`${c * arc} ${c}`} strokeLinecap="round" />
        <circle cx="70" cy="70" r={r} fill="none" stroke={color} strokeWidth="9"
          strokeDasharray={`${c * arc * v} ${c}`} strokeLinecap="round" style={{ transition: "stroke-dasharray .6s ease" }} />
      </svg>
      {tAngle !== null && (
        <div className="absolute inset-0" style={{ transform: `rotate(${tAngle}deg)` }}>
          <div className="absolute left-1/2 top-[6px] h-4 w-[3px] -translate-x-1/2 rounded bg-ink/60" title="alert threshold" />
        </div>
      )}
      <div className="relative text-center">
        <div className={`${small ? "text-2xl" : "text-3xl"} font-extrabold tabular`} style={{ color }}>{Math.round(v * 100)}%</div>
        <div className="text-[10px] font-semibold uppercase tracking-wider text-faint">{label}</div>
      </div>
    </div>
  );
}

export function Segmented<T extends string>({ options, value, onChange }: {
  options: { value: T; label: string }[]; value: T; onChange: (v: T) => void;
}) {
  return (
    <div className="neu-inset-sm inline-flex gap-1 p-1">
      {options.map((o) => (
        <button key={o.value} onClick={() => onChange(o.value)}
          className={`rounded-[10px] px-3 py-1.5 text-xs font-semibold transition ${value === o.value ? "neu-sm text-accent" : "text-muted hover:text-ink"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Loading({ h = 120 }: { h?: number }) {
  return <div className="neu-inset shimmer" style={{ height: h }} />;
}

export function ErrorBox({ msg }: { msg: string }) {
  return (
    <div className="neu-inset p-4 text-sm text-sev-red">
      Could not reach the API: {msg}. Is the backend running on port 8000?
    </div>
  );
}
