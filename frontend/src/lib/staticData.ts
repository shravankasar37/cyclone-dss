/**
 * Static "replay edition": serves the same API paths from pre-computed files in /public/data,
 * produced by scripts/export_static.py from the real models. Enabled with NEXT_PUBLIC_DATA_MODE=static.
 */
export const STATIC_MODE = process.env.NEXT_PUBLIC_DATA_MODE === "static";

type Bundle = {
  track: unknown;
  ri_timeline: unknown;
  times: Record<string, { analysis: unknown; impact: unknown; cap: string; bulletin: unknown }>;
};

const bundles = new Map<string, Promise<Bundle>>();

async function file<T>(rel: string): Promise<T> {
  const r = await fetch(`/data/${rel}`);
  if (!r.ok) throw Object.assign(new Error(`${r.status} /data/${rel}`), { http: true });
  return r.json();
}

function stormBundle(sid: string) {
  if (!bundles.has(sid)) bundles.set(sid, file<Bundle>(`storms/${sid}.json`));
  return bundles.get(sid)!;
}

async function at(sid: string, time: string | null) {
  const b = await stormBundle(sid);
  const keys = Object.keys(b.times);
  const key = time && b.times[time] ? time : time
    ? keys.reduce((best, k) => (Math.abs(Date.parse(k + "Z") - Date.parse(time + "Z")) < Math.abs(Date.parse(best + "Z") - Date.parse(time + "Z")) ? k : best), keys[0])
    : keys[Math.floor(keys.length / 2)];
  return b.times[key];
}

/** Keep only the CAP <info> blocks for the selected languages (the export holds all three). */
function filterCap(xml: string, langs: string[]) {
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  const want = new Set(langs.map((l) => `${l}-IN`));
  Array.from(doc.getElementsByTagName("info")).forEach((info) => {
    const lang = info.getElementsByTagName("language")[0]?.textContent ?? "";
    if (!want.has(lang)) info.parentNode?.removeChild(info);
  });
  return new XMLSerializer().serializeToString(doc);
}

/** Resolve an API path to its pre-computed file. */
export async function staticGet(path: string): Promise<unknown> {
  const u = new URL(path, "http://x");
  const p = u.pathname, q = u.searchParams;
  let m: RegExpMatchArray | null;
  if (p === "/api/storms") return file("storms.json");
  if ((m = p.match(/^\/api\/storms\/([^/]+)\/track$/))) return (await stormBundle(m[1])).track;
  if ((m = p.match(/^\/api\/storms\/([^/]+)\/ri-timeline$/))) return (await stormBundle(m[1])).ri_timeline;
  if ((m = p.match(/^\/api\/storms\/([^/]+)\/analysis$/))) return (await at(m[1], q.get("time"))).analysis;
  if ((m = p.match(/^\/api\/storms\/([^/]+)\/impact$/))) return (await at(m[1], q.get("time"))).impact;
  if ((m = p.match(/^\/api\/storms\/([^/]+)\/bulletin$/))) return (await at(m[1], q.get("time"))).bulletin;
  if ((m = p.match(/^\/api\/storms\/([^/]+)\/cap\.xml$/))) return filterCap((await at(m[1], q.get("time"))).cap, (q.get("langs") ?? "en,hi,gu").split(","));
  if (p === "/api/satellite/storms") return file("sat/storms.json");
  if (p === "/api/satellite/frames") return (await file<{ frames: unknown }>(`sat/${q.get("tcir_sid")}.json`)).frames;
  if (p === "/api/satellite/series") return (await file<{ series: unknown }>(`sat/${q.get("tcir_sid")}.json`)).series;
  if ((m = p.match(/^\/api\/satellite\/frames\/(\d+)$/))) return file(`sat/frame/${m[1]}.json`);
  if (p === "/api/basin/storms") return file("basin/storms.json");
  if ((m = p.match(/^\/api\/basin\/storms\/([^/]+)$/))) return file(`basin/${m[1]}.json`);
  if ((m = p.match(/^\/api\/basin\/scene\/(\d+)$/))) return file(`basin/scene/${m[1]}.json`);
  if (p === "/api/metrics") return file("metrics.json");
  throw Object.assign(new Error(`not available in the online replay edition: ${p}`), { http: true });
}

/** URL for a binary asset that the API would otherwise serve. */
export function assetUrl(apiPath: string, api: string) {
  if (!STATIC_MODE) return `${api}${apiPath}`;
  if (apiPath === "/api/metrics/intensity_eval.png") return "/data/intensity_eval.png";
  return apiPath;
}
