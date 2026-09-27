import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // hide the dev-mode badge: it overlaps the console layout during demos
  devIndicators: false,
};

export default nextConfig;
