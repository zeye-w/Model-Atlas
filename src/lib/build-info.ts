/** Short commit SHA inlined at build time (see /build-info.ts). */
export const buildSha = process.env.NEXT_PUBLIC_BUILD_SHA || "dev";

export const buildCommitUrl = buildSha === "dev"
  ? "https://github.com/zeye-w/Model-Atlas"
  : `https://github.com/zeye-w/Model-Atlas/commit/${buildSha}`;
