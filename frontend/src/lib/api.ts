import { STATIC_MODE, staticGet } from "./staticData";

export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export { STATIC_MODE };

/** GET JSON; retries network failures (API still starting) with backoff for ~40 s before giving up. */
export async function get<T>(path: string, retries = 8): Promise<T> {
  if (STATIC_MODE) return (await staticGet(path)) as T;
  for (let attempt = 0; ; attempt++) {
    try {
      const r = await fetch(`${API}${path}`, { cache: "no-store" });
      if (!r.ok) throw Object.assign(new Error(`${r.status} ${await r.text()}`), { http: true });
      return await r.json();
    } catch (e) {
      if ((e as { http?: boolean }).http || attempt >= retries) throw e;
      await new Promise((res) => setTimeout(res, Math.min(1000 * 2 ** attempt, 8000)));
    }
  }
}

export async function getText(path: string): Promise<string> {
  if (STATIC_MODE) return (await staticGet(path)) as string;
  const r = await fetch(`${API}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.text();
}

export type ClassCode = "D" | "DD" | "CS" | "SCS" | "VSCS" | "ESCS" | "SuCS";
export type Severity = "Red" | "Orange" | "Yellow";

export interface StormSummary {
  sid: string; name: string; season: number; subbasin: string; start: string; end: string;
  peak_kt: number | null; peak_imd_kt: number | null; ri: boolean; peak_class: ClassCode;
}

export interface TrackPoint {
  time: string; lat: number; lon: number; wind_kt: number | null; wind_imd_kt: number | null;
  imd_class: ClassCode; forecastable: boolean;
}
export interface Track { sid: string; name: string; season: number; subbasin: string; points: TrackPoint[] }

export interface Lead {
  lead_h: number; lat: number; lon: number; cone_radius_km: number; wind_kt: number; wind_sd_kt: number;
  wind_imd_kt: number; imd_class: ClassCode;
}
export interface Driver { feature: string; value: number | null; impact: number }
export interface Impact {
  name: string; state: string; lang: string; lat: number; lon: number; eta_h: number; stage: string;
  color: Severity; max_wind_imd_kt: number; min_distance_km: number; certainty: string;
}
export interface Analysis {
  sid: string; name: string; time: string;
  current: {
    lat: number; lon: number; wind_kt: number; wind_imd_kt: number; wind_estimated?: boolean; pressure_hpa: number;
    imd_class: { code: ClassCode; name: string; index: number };
    dist2land_km: number | null; motion_kmh: number | null; heading_deg: number | null; trend_24h_kt: number | null;
  };
  forecast: { method: string; leads: Lead[]; cone: [number, number][] };
  ri: {
    probability: number; threshold: number; climatology: number; ratio_to_climatology: number; flag: boolean;
    definition: string; drivers: Driver[]; observed: number | null;
  };
  genesis: { applicable: boolean; reason: string | null; probability: number | null; definition: string; drivers: Driver[] };
  actual: { lead_h: number; lat: number; lon: number; wind_kt: number | null; track_error_km: number; inside_cone: boolean; wind_error_kt: number | null }[];
  impacts: Impact[];
}

export interface RiTimeline {
  sid: string; threshold: number; climatology: number;
  points: { time: string; ri_probability: number; observed_dw24: number | null; wind_kt: number | null; genesis_probability: number | null }[];
}

export interface SatStorm { tcir_sid: string; name: string; ibtracs_sid: string | null; start: string; end: string; frames: number; peak_kt: number; held_out: boolean }
export interface SatFrame { k: number; time: string; lat: number; lon: number; best_track_kt: number }
export interface ImageAnalysis {
  cnn_wind_kt: number; cnn_wind_raw_kt: number; insat_calibrated: boolean; t_number: number; t_number_raw: number;
  dvorak_wind_kt: number; class_probs: Record<ClassCode, number>;
  class_head: { code: ClassCode; name: string }; class_from_wind: { code: ClassCode; name: string };
  class_from_dvorak: { code: ClassCode; name: string }; pressure_est_hpa: number; scene_type: string;
  adt_features: Record<string, number>; gradcam_png: string; image_png: string;
  name?: string; time?: string; best_track_kt?: number; held_out?: boolean;
  seq?: { dv24_kt: number; ri_probability: number; observed_dv24_kt: number | null };
}

export const CLASS_NAMES: Record<ClassCode, string> = {
  D: "Depression", DD: "Deep Depression", CS: "Cyclonic Storm", SCS: "Severe Cyclonic Storm",
  VSCS: "Very Severe Cyclonic Storm", ESCS: "Extremely Severe Cyclonic Storm", SuCS: "Super Cyclonic Storm",
};
export const CLASS_COLORS: Record<ClassCode, string> = {
  D: "#5aa9e6", DD: "#2bb3c0", CS: "#3fbf7f", SCS: "#e0b100", VSCS: "#f2711c", ESCS: "#e5484d", SuCS: "#9b5de5",
};
export const SEV_VAR: Record<Severity, string> = { Red: "var(--red)", Orange: "var(--orange)", Yellow: "var(--yellow)" };

export const FEATURE_LABELS: Record<string, string> = {
  wind: "Current wind", dw6: "6 h wind change", dw12: "12 h wind change", dw24: "24 h wind change",
  max_wind_so_far: "Peak wind so far", age_h: "Storm age", lat: "Latitude", lon: "Longitude",
  u_motion: "Eastward motion", v_motion: "Northward motion", log_dist2land: "Distance to land",
  is_arabian_sea: "Arabian Sea", month_sin: "Season", month_cos: "Season", wind_imd_eff: "Current wind (IMD)",
  sst_c: "Sea-surface temp", shear_ms: "Vertical wind shear", rh600: "Mid-level humidity",
};

export const kt2kmh = (kt: number) => kt * 1.852;
export const fmtTime = (iso: string) =>
  new Date(iso + (iso.endsWith("Z") || iso.includes("+") ? "" : "Z")).toLocaleString("en-GB", {
    day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZone: "UTC",
  }) + " UTC";
export const fmtShort = (iso: string) =>
  new Date(iso + (iso.endsWith("Z") ? "" : "Z")).toLocaleString("en-GB", {
    day: "2-digit", month: "short", hour: "2-digit", timeZone: "UTC",
  });

// ---------- impact ----------
export interface Landfall {
  status: "possible" | "none" | "over_land" | "insufficient_history"; message?: string;
  probability_72h?: number; india_probability?: number; members?: number; country?: string;
  point?: { lat: number; lon: number }; nearest_district?: { name: string; state: string; distance_km: number };
  time_median?: string; window?: [string, string]; hours?: [number, number, number];
  wind_imd_kt?: [number, number, number]; spread_km_80pct?: number; member_points?: [number, number][];
}
export interface SeaArea { area: string; from_h: number; max_wind_kt: number; sea_state: string; action: string; color: Severity; box: [number, number, number, number] }
export interface Sms { lang: string; text: string; encoding: string; chars: number; segments: number }
export interface ImpactData {
  time: string; landfall: Landfall; sea_areas: SeaArea[]; sms: Sms[];
  exposure: { people_in_gale_envelope: number; by_country: Record<string, number>; by_window: Record<string, number>; method: string };
}

// ---------- basin scan ----------
export interface BasinPoint {
  score: number; cls: "cyclone" | "depression"; lat: number; lon: number; k: number; time: string;
  wind_kt: number; wind_imd_kt: number; t_number: number; dvorak_wind_kt: number; imd_class: ClassCode;
  class_probs: Record<ClassCode, number>; seq?: { dv24_kt: number; ri_probability: number };
  truth?: { lat: number; lon: number; wind_imd_kt: number; wind_kt: number | null; error_km: number };
}
export interface BasinTrack { id: number; points: BasinPoint[]; matched_points: number; median_error_km: number | null; confirmed: boolean; matches_any_system: boolean }
export interface BasinStorm {
  summary: {
    storm: string; scenes: number; start: string; end: string; tracks: number; main_track: number | null;
    first_detection?: string; hours_before_depression?: number | null; hours_before_cyclonic_storm?: number | null;
    detection_rate?: number; median_position_error_km?: number | null; intensity_mae_kt?: number | null;
    tracks_real: number; tracks_false: number; confirmed_real: number; confirmed_false: number; wv_missing: boolean;
    best_track?: { time: string; lat: number; lon: number; wind_imd_kt: number }[];
  };
  tracks: BasinTrack[]; detections: Record<string, { score: number; cls: string; lat: number; lon: number }[]>;
  times: string[]; ks: number[];
}
export interface BasinScene { k: number; time: string; storm: string; png: string; bounds: [number, number, number, number] }

export const fmtPeople = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)} M` : n >= 1e3 ? `${Math.round(n / 1e3)} k` : `${Math.round(n)}`);
