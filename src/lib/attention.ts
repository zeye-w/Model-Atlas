/**
 * Single-head scaled dot-product attention on a short sentence.
 *
 *   X = [token embedding | positional encoding]      (n × d_model, d_model = 8)
 *   Q = X W_Q,  K = X W_K,  V = X W_V                 (n × d_k,     d_k = 4)
 *   S = Q Kᵀ / √d_k   (optionally masked so token i only sees j ≤ i)
 *   A = softmax_row(S)
 *   O = A V
 *
 * Weights come from one of three heads. Two are constructed by hand so their
 * behaviour follows from the algebra (a standard tool in interpretability
 * research); one is random, as at initialisation. None is trained.
 */

export const D_TOKEN = 4;          // token-identity part of the embedding
export const D_POS = 4;            // positional-encoding part (two sin/cos pairs)
export const D_MODEL = D_TOKEN + D_POS;
export const D_K = 4;
/** Angular frequencies (radians per position) of the two sin/cos pairs. */
export const FREQS = [1, .5];
export const MAX_TOKENS = 10;

export type Matrix = number[][];
export type HeadId = "previous-token" | "same-word" | "random";

export const heads: Record<HeadId, { name: string; description: string }> = {
  "previous-token": { name: "Previous-token head", description: "Queries rotate each position back by one step, so every word's query best matches the key of the word just before it." },
  "same-word": { name: "Same-word head", description: "Queries and keys both read only the word-identity part of the embedding, so a word's query best matches every occurrence of the same word." },
  "random": { name: "Random head", description: "Seeded random projections, as a head looks before training. The pattern is real computation but carries no learned meaning." },
};

/* ── Small linear-algebra helpers ─────────────────────────────────────── */
export const matmul = (A: Matrix, B: Matrix): Matrix => A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
export const transpose = (A: Matrix): Matrix => A[0].map((_, j) => A.map((row) => row[j]));
export const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
export const zeros = (r: number, c: number): Matrix => Array.from({ length: r }, () => new Array(c).fill(0));

export function softmax(row: number[]) {
  const finite = row.filter(Number.isFinite);
  const m = Math.max(...finite);
  const e = row.map((v) => (Number.isFinite(v) ? Math.exp(v - m) : 0));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

function random(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** FNV-1a hash, so the same word always gets the same embedding. */
function hash(text: string) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}

export function tokenize(sentence: string) {
  return sentence.toLowerCase().replace(/[^\p{L}\p{N}'\s-]/gu, " ").split(/\s+/).filter(Boolean).slice(0, MAX_TOKENS);
}

/** Unit-length, word-specific vector (deterministic). */
export function tokenEmbedding(word: string) {
  const rng = random(hash(word));
  const v = Array.from({ length: D_TOKEN }, () => 2 * rng() - 1);
  const norm = Math.hypot(...v) || 1;
  return v.map((x) => x / norm);
}

/** Sinusoidal positional encoding with two frequencies, scaled to unit length. */
export function positionalEncoding(pos: number) {
  return FREQS.flatMap((w) => [Math.sin(w * pos), Math.cos(w * pos)]).map((v) => v / Math.SQRT2);
}

export function embed(tokens: string[]): Matrix {
  return tokens.map((t, i) => [...tokenEmbedding(t), ...positionalEncoding(i)]);
}

/* ── Heads ────────────────────────────────────────────────────────────── */
export interface HeadWeights { wq: Matrix; wk: Matrix; wv: Matrix }

/** d_model × d_k matrix that copies the token part (rows 0–3) with factor c. */
function selectToken(c: number): Matrix {
  return Array.from({ length: D_MODEL }, (_, r) => Array.from({ length: D_K }, (_, j) => (r === j ? c : 0)));
}

/**
 * Positional rows rotated by −ω per pair: for a row vector p(i), p(i)·R equals
 * p(i − 1), because [sin ωi, cos ωi] rotated by −ω is [sin ω(i−1), cos ω(i−1)].
 */
function rotatePositionBack(c: number): Matrix {
  const m = zeros(D_MODEL, D_K);
  FREQS.forEach((w, f) => {
    const r = D_TOKEN + 2 * f, cs = Math.cos(w), sn = Math.sin(w);
    // [s, c]·[[cos, sin], [−sin, cos]]... written out for row vectors:
    // s' = s cos ω − c sin ω,  c' = s sin ω + c cos ω
    m[r][2 * f] = c * cs; m[r][2 * f + 1] = c * sn;
    m[r + 1][2 * f] = -c * sn; m[r + 1][2 * f + 1] = c * cs;
  });
  return m;
}

function selectPosition(c: number): Matrix {
  const m = zeros(D_MODEL, D_K);
  for (let j = 0; j < D_K; j++) m[D_TOKEN + j][j] = c;
  return m;
}

export function headWeights(head: HeadId, seed = 1): HeadWeights {
  // Scale factors set how sharp the softmax is. Neighbouring positions differ
  // by only ~0.29 in p(i)·p(j), so the positional head needs a larger factor.
  const sharp = 3.2, positional = 5;
  switch (head) {
    case "previous-token": return { wq: rotatePositionBack(positional), wk: selectPosition(positional), wv: selectToken(1) };
    case "same-word": return { wq: selectToken(sharp), wk: selectToken(sharp), wv: selectToken(1) };
    case "random": {
      const rng = random(seed ^ 0xa77e);
      const g = () => { const u = Math.max(rng(), 1e-12), v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
      const make = () => Array.from({ length: D_MODEL }, () => Array.from({ length: D_K }, () => g() / Math.sqrt(D_MODEL) * 1.6));
      return { wq: make(), wk: make(), wv: make() };
    }
  }
}

/* ── Forward pass ─────────────────────────────────────────────────────── */
export interface AttentionOptions { causal: boolean; scale: boolean }
export interface AttentionTrace {
  tokens: string[]; X: Matrix; Q: Matrix; K: Matrix; V: Matrix;
  /** Raw scores Q Kᵀ (before scaling). */
  raw: Matrix;
  /** Scaled (and masked) scores: −Infinity where masked. */
  scores: Matrix;
  A: Matrix; O: Matrix;
  /** Cosine similarity of each output row to each token's value vector. */
  similarity: Matrix;
}

export function attention(tokens: string[], w: HeadWeights, opt: AttentionOptions): AttentionTrace {
  const X = embed(tokens);
  const Q = matmul(X, w.wq), K = matmul(X, w.wk), V = matmul(X, w.wv);
  const raw = matmul(Q, transpose(K));
  const factor = opt.scale ? 1 / Math.sqrt(D_K) : 1;
  const scores = raw.map((row, i) => row.map((s, j) => (opt.causal && j > i ? -Infinity : s * factor)));
  const A = scores.map(softmax);
  const O = matmul(A, V);
  const cos = (a: number[], b: number[]) => { const d = Math.hypot(...a) * Math.hypot(...b); return d ? dot(a, b) / d : 0; };
  const similarity = O.map((o) => V.map((v) => cos(o, v)));
  return { tokens, X, Q, K, V, raw, scores, A, O, similarity };
}

export const phases = [
  { id: "embed", title: "Embed" },
  { id: "project", title: "Queries, keys, values" },
  { id: "scores", title: "Scores" },
  { id: "softmax", title: "Softmax" },
  { id: "output", title: "Weighted sum" },
] as const;
export type AttentionPhase = typeof phases[number]["id"];
