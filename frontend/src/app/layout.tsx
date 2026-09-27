import type { Metadata } from "next";
import { JetBrains_Mono, Manrope } from "next/font/google";
import { Nav } from "@/components/Nav";
import "mapbox-gl/dist/mapbox-gl.css";
import "./globals.css";

const ui = Manrope({ variable: "--font-ui", subsets: ["latin"] });
const code = JetBrains_Mono({ variable: "--font-code", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Cyclone Decision Support · PS 26070",
  description: "AI-assisted tropical cyclone detection, intensity, rapid-intensification and track guidance for the North Indian Ocean (research prototype).",
};

const themeScript = `try{var t=localStorage.getItem('theme');if(!t)t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${ui.variable} ${code.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-screen">
        <Nav />
        <main className="mx-auto w-full max-w-[1920px] px-4 pb-4 pt-2 sm:px-5">{children}</main>
      </body>
    </html>
  );
}
