import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export: builds to `out/`, servable by any host with no Node runtime.
  output: "export",
  images: { unoptimized: true },
  // JSON in public/ is imported directly rather than fetched, so the page has
  // no loading state and no failure mode -- the data ships with the bundle.
  reactStrictMode: true,
};

export default nextConfig;
