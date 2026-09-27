"use client";
import { useEffect, useRef, useState } from "react";
import type { GeoJSONSource, ImageSource, Map as MapboxMap } from "mapbox-gl";
import type { BasinScene, BasinStorm } from "@/lib/api";

const TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN ?? "";
const STYLE = { light: "mapbox://styles/mapbox/light-v11", dark: "mapbox://styles/mapbox/dark-v11" };
type FC = GeoJSON.FeatureCollection;
const empty: FC = { type: "FeatureCollection", features: [] };
const BLANK = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

function corners(b: [number, number, number, number]): [[number, number], [number, number], [number, number], [number, number]] {
  return [[b[0], b[3]], [b[2], b[3]], [b[2], b[1]], [b[0], b[1]]];
}

function build(storm: BasinStorm | null, k: number, selected: number | null) {
  if (!storm) return { best: empty, bestNow: empty, auto: empty, dets: empty };
  const t = storm.times[storm.ks.indexOf(k)];
  const bt = storm.summary.best_track ?? [];
  const line = (c: [number, number][], props = {}): GeoJSON.Feature => ({ type: "Feature", properties: props, geometry: { type: "LineString", coordinates: c } });
  const auto = storm.tracks.map((tr) => {
    const pts = tr.points.filter((p) => p.k <= k);
    return pts.length > 1 ? line(pts.map((p) => [p.lon, p.lat]), { main: tr.id === storm.summary.main_track ? 1 : 0, confirmed: tr.confirmed ? 1 : 0 }) : null;
  }).filter(Boolean) as GeoJSON.Feature[];
  const now = storm.tracks.flatMap((tr) => tr.points.filter((p) => p.k === k).map((p) => ({ ...p, track: tr.id, confirmed: tr.confirmed })));
  const bestNow = bt.filter((p) => p.time === t);
  return {
    best: { type: "FeatureCollection", features: bt.length > 1 ? [line(bt.map((p) => [p.lon, p.lat]))] : [] } as FC,
    bestNow: { type: "FeatureCollection", features: bestNow.map((p) => ({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [p.lon, p.lat] } })) } as FC,
    auto: { type: "FeatureCollection", features: auto } as FC,
    dets: { type: "FeatureCollection", features: now.map((p) => ({ type: "Feature",
      properties: { cls: p.cls, label: p.confirmed ? `${Math.round(p.wind_imd_kt)} kt` : "", sel: p.track === selected ? 1 : 0, track: p.track, confirmed: p.confirmed ? 1 : 0 },
      geometry: { type: "Point", coordinates: [p.lon, p.lat] } })) } as FC,
  };
}

export default function BasinMap({ storm, scene, k, selected, onSelect }: {
  storm: BasinStorm | null; scene: BasinScene | null; k: number; selected: number | null; onSelect: (track: number) => void;
}) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<MapboxMap | null>(null);
  const [ready, setReady] = useState(0);
  const onSelectRef = useRef(onSelect);
  useEffect(() => { onSelectRef.current = onSelect; }, [onSelect]);

  useEffect(() => {
    if (!el.current || !TOKEN) return;
    let disposed = false;
    import("mapbox-gl").then(({ default: mapboxgl }) => {
      if (disposed || !el.current) return;
      mapboxgl.accessToken = TOKEN;
      const theme = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
      const m = new mapboxgl.Map({ container: el.current, style: STYLE[theme], center: [72, 15], zoom: 3.3, projection: "mercator" });
      m.addControl(new mapboxgl.NavigationControl({ showCompass: false }), "top-right");
      map.current = m;
      m.on("style.load", () => { addLayers(m); setReady((r) => r + 1); });
      m.on("click", "dets", (e) => { const f = e.features?.[0]; if (f) onSelectRef.current(Number((f as unknown as { properties: { track: number } }).properties.track)); });
      m.on("mouseenter", "dets", () => (m.getCanvas().style.cursor = "pointer"));
      m.on("mouseleave", "dets", () => (m.getCanvas().style.cursor = ""));
    });
    const onTheme = (e: Event) => map.current?.setStyle(STYLE[(e as CustomEvent).detail as "light" | "dark"]);
    window.addEventListener("themechange", onTheme);
    return () => { disposed = true; window.removeEventListener("themechange", onTheme); map.current?.remove(); map.current = null; };
  }, []);

  useEffect(() => {
    const m = map.current;
    if (!m || !ready || !m.getSource("best")) return;
    if (scene) (m.getSource("ir") as ImageSource).updateImage({ url: scene.png, coordinates: corners(scene.bounds) });
    const d = build(storm, k, selected);
    (Object.keys(d) as (keyof typeof d)[]).forEach((key) => (m.getSource(key) as GeoJSONSource).setData(d[key]));
  }, [storm, scene, k, selected, ready]);

  if (!TOKEN) return <div className="neu-inset grid h-full place-items-center p-6 text-sm text-muted">Add NEXT_PUBLIC_MAPBOX_TOKEN to show the map.</div>;
  return (
    <div className="absolute inset-1.5 overflow-hidden rounded-[18px]">
      <div ref={el} className="h-full w-full" />
    </div>
  );
}

function addLayers(m: MapboxMap) {
  m.addSource("ir", { type: "image", url: BLANK, coordinates: corners([40, 0, 100, 30]) });
  const firstSymbol = m.getStyle().layers?.find((l) => l.type === "symbol")?.id;
  m.addLayer({ id: "ir", type: "raster", source: "ir", paint: { "raster-opacity": 0.9, "raster-fade-duration": 0 } }, firstSymbol);
  ["best", "bestNow", "auto", "dets"].forEach((id) => m.addSource(id, { type: "geojson", data: empty }));
  m.addLayer({ id: "best", type: "line", source: "best", paint: { "line-color": "#6a778b", "line-width": 2, "line-dasharray": [1.5, 1.5] } });
  m.addLayer({ id: "auto", type: "line", source: "auto", paint: {
    "line-color": ["case", ["==", ["get", "main"], 1], "#4a6cf7", "#9aa6b8"],
    "line-width": ["case", ["==", ["get", "confirmed"], 1], 3, 1.5],
    "line-opacity": ["case", ["==", ["get", "confirmed"], 1], 1, 0.55] } });
  m.addLayer({ id: "bestNow", type: "circle", source: "bestNow", paint: { "circle-radius": 9, "circle-color": "rgba(0,0,0,0)", "circle-stroke-width": 2, "circle-stroke-color": "#6a778b" } });
  m.addLayer({ id: "dets", type: "circle", source: "dets", paint: {
    "circle-radius": ["case", ["==", ["get", "sel"], 1], 9, ["==", ["get", "confirmed"], 1], 7, 4.5],
    "circle-opacity": ["case", ["==", ["get", "confirmed"], 1], 1, 0.6],
    "circle-color": ["match", ["get", "cls"], "cyclone", "#e5484d", "#e0a100"],
    "circle-stroke-width": ["case", ["==", ["get", "sel"], 1], 4, 2], "circle-stroke-color": "#ffffff" } });
  m.addLayer({ id: "det-labels", type: "symbol", source: "dets", layout: { "text-field": ["get", "label"], "text-size": 11, "text-offset": [0, 1.5], "text-font": ["DIN Pro Bold", "Arial Unicode MS Bold"] },
    paint: { "text-color": "#2b3445", "text-halo-color": "#ffffff", "text-halo-width": 1.5 } });
}
