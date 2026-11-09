import type { NextConfig } from "next";
import { applyBuildEnv } from "./build-info";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  env: { NEXT_PUBLIC_BUILD_SHA: applyBuildEnv() },
};

export default nextConfig;
