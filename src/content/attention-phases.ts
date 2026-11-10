import type { AttentionPhase } from "@/lib/attention";

/** Explanations for each step of the attention laboratory. Formulas render on the server. */
export const attentionPhaseText: Record<AttentionPhase, { latex: string; summary: string }> = {
  "embed": {
    latex: String.raw`\mathbf{x}_i = \big[\, \mathbf{e}(\text{word}_i) \;\big|\; \mathbf{p}(i) \,\big] \in \mathbb{R}^{8}`,
    summary: "Each word becomes a vector: four numbers that identify the word, followed by four that encode its position as sines and cosines at two frequencies. Identical words share the first half; the second half differs by position.",
  },
  "project": {
    latex: String.raw`Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V \qquad (W \in \mathbb{R}^{8 \times 4})`,
    summary: "Three learned matrices give every word a query (what it is looking for), a key (what it offers to be found by) and a value (what it passes on if chosen).",
  },
  "scores": {
    latex: String.raw`S_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} \qquad \big(S_{ij} = -\infty \text{ for } j > i \text{ when causal}\big)`,
    summary: "Every query is compared with every key by a dot product. Dividing by √d_k keeps the scores in a range where softmax is not saturated; the causal mask stops a word from looking ahead.",
  },
  "softmax": {
    latex: String.raw`A_{ij} = \frac{e^{S_{ij}}}{\sum_{m} e^{S_{im}}} \qquad \textstyle\sum_j A_{ij} = 1`,
    summary: "Each row of scores becomes a probability distribution: how much the word on that row attends to every other word.",
  },
  "output": {
    latex: String.raw`\mathbf{o}_i = \sum_{j} A_{ij}\, \mathbf{v}_j \qquad O = A V`,
    summary: "The output for each word is the attention-weighted average of the value vectors. A head that attends to the previous word therefore outputs a copy of that word's value.",
  },
};
