"use client";
import { useEffect, useRef, useState } from "react";
import type { Map as MapboxMap, Marker, GeoJSONSource } from "mapbox-gl";
import { CLASS_COLORS, type Analysis, type ImpactData, type TrackPoint } from "@/lib/api";

const TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN ?? "";
const STYLE = { light: "mapbox://styles/mapbox/light-v11", dark: "mapbox://styles/mapbox/dark-v11" };
type FC = GeoJSON.FeatureCollection;
const empty: FC = { type: "FeatureCollection", features: [] };

function currentTheme(): "light" | "dark" {
  return typeof document !== "undefined" && document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function buildData(points: TrackPoint[], idx: number, a: Analysis | null, showActual: boolean, impact: ImpactData | null) {
  const line = (coords: [number, number][]): GeoJSON.Feature => ({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: coords } });
  const past = points.slice(0, idx + 1);
  const future = points.slice(idx);
  const pts: FC = { type: "FeatureCollection", features: past.map((p) => ({ type: "Feature", properties: { color: CLASS_COLORS[p.imd_class], cls: p.imd_class, time: p.time }, geometry: { type: "Point", coordinates: [p.lon, p.lat] } })) };
  const fcLeads = a?.forecast.leads ?? [];
  const cur = a ? [a.current.lon, a.current.lat] as [number, number] : null;
  return {
    past: { type: "FeatureCollection", features: past.length > 1 ? [line(past.map((p) => [p.lon, p.lat]))] : [] } as FC,
    future: { type: "FeatureCollection", features: showActual && future.length > 1 ? [line(future.map((p) => [p.lon, p.lat]))] : [] } as FC,
    pts,
    cone: a ? { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [a.forecast.cone] } }] } as FC : empty,
    fcLine: cur ? { type: "FeatureCollection", features: [line([cur, ...fcLeads.map((l) => [l.lon, l.lat] as [number, number])])] } as FC : empty,
    fcPts: { type: "FeatureCollection", features: fcLeads.map((l) => ({ type: "Feature", properties: { label: `+${l.lead_h}h`, cls: l.imd_class }, geometry: { type: "Point", coordinates: [l.lon, l.lat] } })) } as FC,
    actual: { type: "FeatureCollection", features: showActual ? (a?.actual ?? []).map((x) => ({ type: "Feature", properties: { label: `${Math.round(x.track_error_km)} km` }, geometry: { type: "Point", coordinates: [x.lon, x.lat] } })) : [] } as FC,
    landfall: { type: "FeatureCollection", features: (impact?.landfall.member_points ?? []).map(([lo, la]) => ({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [lo, la] } })) } as FC,
    seas: { type: "FeatureCollection", features: (impact?.sea_areas ?? []).map((s) => ({ type: "Feature", properties: { color: s.color, area: s.area },
      geometry: { type: "Polygon", coordinates: [[[s.box[0], s.box[1]], [s.box[2], s.box[1]], [s.box[2], s.box[3]], [s.box[0], s.box[3]], [s.box[0], s.box[1]]]] } })) } as FC,
    districts: { type: "FeatureCollection", features: (a?.impacts ?? []).map((d) => ({ type: "Feature", properties: { name: d.name, state: d.state, color: d.color, stage: d.stage, eta: d.eta_h }, geometry: { type: "Point", coordinates: [d.lon, d.lat] } })) } as FC,
  };
}

export default function StormMap({ points, idx, analysis, showActual, impact = null }: {
  points: TrackPoint[]; idx: number; analysis: Analysis | null; showActual: boolean; impact?: ImpactData | null;
}) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<MapboxMap | null>(null);
  const marker = useRef<Marker | null>(null);
  const [ready, setReady] = useState(0);
  const sidRef = useRef<string | null>(null);

  useEffect(() => {
    if (!el.current || !TOKEN) return;
    let disposed = false;
    import("mapbox-gl").then(({ default: mapboxgl }) => {
      if (disposed || !el.current) return;
      mapboxgl.accessToken = TOKEN;
      const m = new mapboxgl.Map({ container: el.current, style: STYLE[currentTheme()], center: [80, 15], zoom: 3.6,
        projection: "mercator", attributionControl: true, cooperativeGestures: false });
      m.addControl(new mapboxgl.NavigationControl({ showCompass: false }), "top-right");
      map.current = m;
      const node = document.createElement("div");
      node.className = "storm-pulse";
      marker.current = new mapboxgl.Marker({ element: node });
      m.on("style.load", () => { addLayers(m); setReady((r) => r + 1); });
      m.on("click", "districts", (e) => {
        const f = e.features?.[0];
        if (!f) return;
        const p = (f as unknown as { properties: Record<string, string> }).properties;
        new mapboxgl.Popup({ closeButton: false }).setLngLat(e.lngLat)
          .setHTML(`<b>${p.name}</b><br/>${p.state}<br/>${p.color} · ${p.stage}<br/>closest approach ~${p.eta} h`).addTo(m);
      });
      m.on("mouseenter", "districts", () => (m.getCanvas().style.cursor = "pointer"));
      m.on("mouseleave", "districts", () => (m.getCanvas().style.cursor = ""));
    });
    const onTheme = (e: Event) => map.current?.setStyle(STYLE[(e as CustomEvent).detail as "light" | "dark"]);
    window.addEventListener("themechange", onTheme);
    return () => { disposed = true; window.removeEventListener("themechange", onTheme); map.current?.remove(); map.current = null; };
  }, []);

  // push data whenever inputs change
  useEffect(() => {
    const m = map.current;
    if (!m || !ready || !m.getSource("past")) return;
    const d = buildData(points, idx, analysis, showActual, impact);
    (Object.keys(d) as (keyof typeof d)[]).forEach((k) => (m.getSource(k) as GeoJSONSource | undefined)?.setData(d[k]));
    if (analysis && marker.current) marker.current.setLngLat([analysis.current.lon, analysis.current.lat]).addTo(m);
  }, [points, idx, analysis, showActual, impact, ready]);

  // fit to storm when a new storm is loaded
  useEffect(() => {
    const m = map.current;
    if (!m || !ready || !points.length || !analysis) return;
    if (sidRef.current === analysis.sid) return;
    sidRef.current = analysis.sid;
    const lons = points.map((p) => p.lon).concat(analysis.forecast.cone.map((c) => c[0]));
    const lats = points.map((p) => p.lat).concat(analysis.forecast.cone.map((c) => c[1]));
    m.fitBounds([[Math.min(...lons) - 2, Math.min(...lats) - 2], [Math.max(...lons) + 2, Math.max(...lats) + 2]], { padding: 30, duration: 900 });
  }, [points, analysis, ready]);

  if (!TOKEN) {
    return <div className="neu-inset grid h-full min-h-[420px] place-items-center p-6 text-sm text-muted">
      Add NEXT_PUBLIC_MAPBOX_TOKEN to frontend/.env.local to show the map.
    </div>;
  }
  return (
    <div className="absolute inset-1.5 overflow-hidden rounded-[18px]">
      <div ref={el} className="h-full w-full" />
    </div>
  );
}

function addLayers(m: MapboxMap) {
  const src = (id: string) => { if (!m.getSource(id)) m.addSource(id, { type: "geojson", data: empty }); };
  ["seas", "cone", "past", "future", "pts", "fcLine", "fcPts", "actual", "landfall", "districts"].forEach(src);
  const accent = "#4a6cf7";
  const sev = ["match", ["get", "color"], "Red", "#e5484d", "Orange", "#f2711c", "#e0a100"] as unknown as string;
  m.addLayer({ id: "seas", type: "fill", source: "seas", paint: { "fill-color": sev, "fill-opacity": 0.07 } });
  m.addLayer({ id: "seas-edge", type: "line", source: "seas", paint: { "line-color": sev, "line-width": 1.2, "line-dasharray": [3, 2], "line-opacity": 0.7 } });
  m.addLayer({ id: "cone", type: "fill", source: "cone", paint: { "fill-color": accent, "fill-opacity": 0.16 } });
  m.addLayer({ id: "cone-edge", type: "line", source: "cone", paint: { "line-color": accent, "line-width": 1.4, "line-dasharray": [2, 2], "line-opacity": 0.8 } });
  m.addLayer({ id: "landfall", type: "circle", source: "landfall", paint: { "circle-radius": 3, "circle-color": "#9b5de5", "circle-opacity": 0.75 } });
  m.addLayer({ id: "districts", type: "circle", source: "districts", paint: {
    "circle-radius": 7, "circle-stroke-width": 2, "circle-stroke-color": "#ffffff",
    "circle-color": ["match", ["get", "color"], "Red", "#e5484d", "Orange", "#f2711c", "#e0a100"] } });
  m.addLayer({ id: "future", type: "line", source: "future", paint: { "line-color": "#8a94a6", "line-width": 2.5, "line-dasharray": [1, 1.6] } });
  m.addLayer({ id: "past", type: "line", source: "past", paint: { "line-color": "#6a778b", "line-width": 3 } });
  m.addLayer({ id: "pts", type: "circle", source: "pts", paint: { "circle-radius": 4.5, "circle-color": ["get", "color"], "circle-stroke-width": 1.5, "circle-stroke-color": "#ffffff" } });
  m.addLayer({ id: "fcLine", type: "line", source: "fcLine", paint: { "line-color": accent, "line-width": 3 } });
  m.addLayer({ id: "actual", type: "circle", source: "actual", paint: { "circle-radius": 5, "circle-color": "rgba(0,0,0,0)", "circle-stroke-width": 2, "circle-stroke-color": "#8a94a6" } });
  m.addLayer({ id: "fcPts", type: "circle", source: "fcPts", paint: { "circle-radius": 6, "circle-color": "#ffffff", "circle-stroke-width": 3, "circle-stroke-color": accent } });
  m.addLayer({ id: "fcLabels", type: "symbol", source: "fcPts", layout: { "text-field": ["get", "label"], "text-size": 11, "text-offset": [0, 1.3], "text-font": ["DIN Pro Bold", "Arial Unicode MS Bold"] },
    paint: { "text-color": accent, "text-halo-color": "#ffffff", "text-halo-width": 1.5 } });
}
