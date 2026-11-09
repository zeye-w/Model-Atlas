import { execSync } from "node:child_process";

/**
 * Resolve the commit this build was made from and expose it as
 * NEXT_PUBLIC_BUILD_SHA so both Next.js and vinext inline it at build time.
 *
 * Order: Cloudflare Workers Builds → GitHub Actions → local git → "dev".
 */
export function resolveBuildSha(): string {
  const fromEnv = process.env.NEXT_PUBLIC_BUILD_SHA || process.env.WORKERS_CI_COMMIT_SHA || process.env.GITHUB_SHA;
  if (fromEnv) return fromEnv.slice(0, 7);
  try {
    return execSync("git rev-parse --short=7 HEAD", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim() || "dev";
  } catch {
    return "dev";
  }
}

export function applyBuildEnv(): string {
  const sha = resolveBuildSha();
  process.env.NEXT_PUBLIC_BUILD_SHA = sha;
  return sha;
}
