"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSyncExternalStore } from "react";
import { Moon, Sun, Tornado } from "lucide-react";

const LINKS = [
  { href: "/", label: "Operations" },
  { href: "/basin", label: "Basin scan" },
  { href: "/satellite", label: "Satellite analysis" },
  { href: "/validation", label: "Validation" },
  { href: "/about", label: "About & limits" },
];

export function Nav() {
  const path = usePathname();
  // the inline script in layout.tsx sets data-theme before paint; this just mirrors it
  const theme = useSyncExternalStore(
    (cb) => { window.addEventListener("themechange", cb); return () => window.removeEventListener("themechange", cb); },
    () => (document.documentElement.dataset.theme === "dark" ? "dark" : "light"),
    () => "light",
  );
  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch {}
    window.dispatchEvent(new CustomEvent("themechange", { detail: next }));
  };
  return (
    <header className="mx-auto flex w-full max-w-[1920px] flex-wrap items-center gap-4 px-4 py-3 sm:px-5 xl:h-16 xl:flex-nowrap xl:py-0">
      <Link href="/" className="flex items-center gap-3">
        <span className="neu-sm grid h-10 w-10 place-items-center text-accent"><Tornado size={20} /></span>
        <span>
          <span className="block text-[15px] font-extrabold leading-tight text-ink">Cyclone Decision Support</span>
          <span className="block text-[11px] font-semibold text-faint">SIH 2026 · PS 26070 · North Indian Ocean</span>
        </span>
      </Link>
      <nav className="neu-inset-sm order-3 flex w-full gap-1 overflow-x-auto p-1 md:order-none md:ml-6 md:w-auto">
        {LINKS.map((l) => (
          <Link key={l.href} href={l.href}
            className={`whitespace-nowrap rounded-[10px] px-3.5 py-2 text-xs font-semibold transition ${path === l.href ? "neu-sm text-accent" : "text-muted hover:text-ink"}`}>
            {l.label}
          </Link>
        ))}
      </nav>
      <div className="ml-auto flex items-center gap-3">
        <button onClick={toggle} className="neu-btn grid h-10 w-10 place-items-center" aria-label="Toggle theme">
          {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
        </button>
      </div>
    </header>
  );
}
