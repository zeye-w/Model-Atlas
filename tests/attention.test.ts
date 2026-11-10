import assert from "node:assert/strict";
import test from "node:test";
import { D_K, MAX_TOKENS, attention, dot, embed, headWeights, matmul, positionalEncoding, softmax, tokenEmbedding, tokenize, transpose } from "../src/lib/attention";

const near = (a: number, b: number, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `expected ${b}, got ${a}`);
const argmax = (row: number[]) => row.indexOf(Math.max(...row));
const sentences = [
  "the cat sat on the mat",
  "a b c d e f g h i j",
  "attention is all you need",
  "to be or not to be",
];

test("tokenizer lower-cases, strips punctuation and caps the length", () => {
  assert.deepEqual(tokenize("The cat, sat!  On the MAT."), ["the", "cat", "sat", "on", "the", "mat"]);
  assert.equal(tokenize("w ".repeat(30)).length, MAX_TOKENS);
});

test("embeddings: same word → same vector; unit-length parts", () => {
  assert.deepEqual(tokenEmbedding("cat"), tokenEmbedding("cat"));
  assert.notDeepEqual(tokenEmbedding("cat"), tokenEmbedding("mat"));
  near(Math.hypot(...tokenEmbedding("anything")), 1);
  near(Math.hypot(...positionalEncoding(7)), 1);
});

test("softmax handles masked (−∞) entries and sums to one", () => {
  const p = softmax([1, -Infinity, 1]);
  assert.deepEqual(p, [.5, 0, .5]);
});

test("forward pass matches the definition: S = QKᵀ/√d_k, rows of A sum to 1, O = AV", () => {
  const tokens = tokenize(sentences[0]);
  const w = headWeights("random", 3);
  const t = attention(tokens, w, { causal: false, scale: true });
  const X = embed(tokens);
  assert.deepEqual(t.Q, matmul(X, w.wq));
  t.raw.forEach((row, i) => row.forEach((s, j) => { near(s, dot(t.Q[i], t.K[j])); near(t.scores[i][j], s / Math.sqrt(D_K)); }));
  t.A.forEach((row) => near(row.reduce((a, b) => a + b, 0), 1));
  assert.deepEqual(t.O, matmul(t.A, t.V));
  assert.deepEqual(t.raw, matmul(t.Q, transpose(t.K)));
});

test("causal mask: no attention to later tokens; first token attends only to itself", () => {
  const t = attention(tokenize(sentences[2]), headWeights("random", 9), { causal: true, scale: true });
  t.A.forEach((row, i) => row.forEach((a, j) => { if (j > i) assert.equal(a, 0); }));
  near(t.A[0][0], 1);
});

test("turning scaling off sharpens attention (larger logits)", () => {
  const tokens = tokenize(sentences[0]);
  const scaled = attention(tokens, headWeights("random", 4), { causal: false, scale: true });
  const unscaled = attention(tokens, headWeights("random", 4), { causal: false, scale: false });
  const peak = (A: number[][]) => A.reduce((s, row) => s + Math.max(...row), 0);
  assert.ok(peak(unscaled.A) > peak(scaled.A));
});

test("previous-token head: every word i ≥ 1 attends most to word i − 1", () => {
  for (const s of sentences) for (const causal of [true, false]) {
    const t = attention(tokenize(s), headWeights("previous-token"), { causal, scale: true });
    for (let i = 1; i < t.tokens.length; i++) {
      assert.equal(argmax(t.A[i]), i - 1, `"${s}" token ${i}`);
      assert.ok(t.A[i][i - 1] > .5, `"${s}" token ${i}: weight ${t.A[i][i - 1]}`);
    }
  }
});

test("same-word head: a repeated word attends mostly to its own occurrences", () => {
  const tokens = tokenize("to be or not to be");
  const t = attention(tokens, headWeights("same-word"), { causal: false, scale: true });
  tokens.forEach((word, i) => {
    const same = tokens.reduce((s, w, j) => s + (w === word ? t.A[i][j] : 0), 0);
    assert.ok(same > .6, `${word}@${i}: ${same}`);
    assert.equal(tokens[argmax(t.A[i])], word);
  });
});

test("previous-token head copies the previous word: its output is most similar to that word's value", () => {
  const t = attention(tokenize(sentences[0]), headWeights("previous-token"), { causal: true, scale: true });
  for (let i = 1; i < t.tokens.length; i++) {
    const best = t.similarity[i].reduce((b, v, j) => (v > t.similarity[i][b] ? j : b), 0);
    assert.equal(t.tokens[best], t.tokens[i - 1]);
  }
});
