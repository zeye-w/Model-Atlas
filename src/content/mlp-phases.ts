/**
 * Explanations for each animation phase, keyed by `phaseTextKey`
 * ("act-…" and "backprop-…" depend on the activation). Formulas are LaTeX,
 * rendered on the server; ℓ is the hidden layer the phase works on.
 */
export const mlpPhaseText: Record<string, { latex: string; summary: string }> = {
  "input": {
    latex: String.raw`\mathbf{a}^{(0)} = \mathbf{x} = (x_1,\ x_2)`,
    summary: "One training point enters the network. Its two coordinates become the input activations.",
  },
  "sum": {
    latex: String.raw`z^{(\ell)}_j = \sum_{i} W^{(\ell)}_{ji}\, a^{(\ell-1)}_i + b^{(\ell)}_j`,
    summary: "Each unit in this layer multiplies every activation from the layer before by a weight, adds them up, and adds its bias.",
  },
  "act-tanh": {
    latex: String.raw`a^{(\ell)}_j = \tanh\!\big(z^{(\ell)}_j\big) \in (-1, 1)`,
    summary: "tanh squashes each sum into (−1, 1). It is zero-centred and its slope is 1 at zero, so gradients pass through well while units are not saturated.",
  },
  "act-sigmoid": {
    latex: String.raw`a^{(\ell)}_j = \sigma\big(z^{(\ell)}_j\big) = \frac{1}{1 + e^{-z^{(\ell)}_j}} \in (0, 1)`,
    summary: "The logistic sigmoid squashes each sum into (0, 1). Its slope is at most 0.25, which matters on the way back.",
  },
  "act-relu": {
    latex: String.raw`a^{(\ell)}_j = \max\big(0,\ z^{(\ell)}_j\big)`,
    summary: "ReLU passes positive sums unchanged and zeroes negative ones. A unit whose sum is negative for every input is “dead”: it outputs zero and receives no gradient.",
  },
  "output-sum": {
    latex: String.raw`z_k = \sum_{j} W^{(L+1)}_{kj}\, a^{(L)}_j + b^{(L+1)}_k`,
    summary: "The output layer takes a weighted sum of the last hidden layer, producing one score (logit) per class.",
  },
  "softmax": {
    latex: String.raw`p_k = \frac{e^{z_k}}{\sum_m e^{z_m}} \qquad L = -\ln p_y`,
    summary: "Softmax turns the scores into probabilities. The loss is small when the true class gets high probability.",
  },
  "output-grad": {
    latex: String.raw`\delta^{(L+1)}_k = p_k - [k = y] \qquad \frac{\partial L}{\partial W^{(L+1)}_{kj}} = \delta^{(L+1)}_k\, a^{(L)}_j`,
    summary: "Backward pass begins. For softmax with cross-entropy, the error at each output is predicted minus target. Each output weight's gradient is that error times the activation it carried.",
  },
  "backprop-tanh": {
    latex: String.raw`\delta^{(\ell)}_j = \Big(\sum_k W^{(\ell+1)}_{kj}\, \delta^{(\ell+1)}_k\Big)\big(1 - (a^{(\ell)}_j)^2\big) \qquad \frac{\partial L}{\partial W^{(\ell)}_{ji}} = \delta^{(\ell)}_j\, a^{(\ell-1)}_i`,
    summary: "The chain rule sends the errors of the layer above back along the same weights, scaled by tanh's slope 1 − a², which is close to 1 for unsaturated units.",
  },
  "backprop-sigmoid": {
    latex: String.raw`\delta^{(\ell)}_j = \Big(\sum_k W^{(\ell+1)}_{kj}\, \delta^{(\ell+1)}_k\Big)\, a^{(\ell)}_j\big(1 - a^{(\ell)}_j\big) \qquad \frac{\partial L}{\partial W^{(\ell)}_{ji}} = \delta^{(\ell)}_j\, a^{(\ell-1)}_i`,
    summary: "The sigmoid's slope a(1 − a) is at most 0.25, so every layer the error passes through shrinks it by a factor of four or more. With several layers, the first ones barely learn — the vanishing-gradient problem.",
  },
  "backprop-relu": {
    latex: String.raw`\delta^{(\ell)}_j = \Big(\sum_k W^{(\ell+1)}_{kj}\, \delta^{(\ell+1)}_k\Big)\,\big[z^{(\ell)}_j > 0\big] \qquad \frac{\partial L}{\partial W^{(\ell)}_{ji}} = \delta^{(\ell)}_j\, a^{(\ell-1)}_i`,
    summary: "ReLU's slope is exactly 1 for active units and 0 for inactive ones, so errors pass back undiminished through active units and not at all through the rest.",
  },
  "update": {
    latex: String.raw`\theta \leftarrow \theta - \eta\, \frac{\partial L}{\partial \theta}`,
    summary: "Every weight and bias takes a small step against its gradient. Edge thickness and colour change to the new values.",
  },
};
