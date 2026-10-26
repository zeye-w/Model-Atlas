import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Static export: Cloudflare Workers serves the generated `out/` directory as static assets.
  output: "export",
};

export default nextConfig;
