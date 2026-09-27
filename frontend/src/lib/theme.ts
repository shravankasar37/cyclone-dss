"use client";
import { useEffect, useState } from "react";

const KEYS = ["--text", "--muted", "--faint", "--accent", "--red", "--orange", "--yellow", "--green", "--sd", "--bg"] as const;
export type Palette = Record<(typeof KEYS)[number], string>;

function read(): Palette {
  const s = getComputedStyle(document.documentElement);
  return Object.fromEntries(KEYS.map((k) => [k, s.getPropertyValue(k).trim()])) as Palette;
}

/** Resolved theme colours for chart libraries that cannot use CSS variables. */
export function usePalette(): Palette | null {
  const [p, setP] = useState<Palette | null>(null);
  useEffect(() => {
    const update = () => requestAnimationFrame(() => setP(read()));
    update();
    window.addEventListener("themechange", update);
    return () => window.removeEventListener("themechange", update);
  }, []);
  return p;
}
