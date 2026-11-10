import type { PhaseId } from "@/lib/mlp";

/** Explanations for each animation phase. Formulas are LaTeX, rendered on the server. */
export const mlpPhaseText: Record<PhaseId, { latex: string; summary: string }> = {
  "input": {
    latex: String.raw`\mathbf{x} = (x_1,\ x_2)`,
    summary: "One training point enters the network. Its two coordinates become the input activations.",
  },
  "hidden-sum": {
    latex: String.raw`z^{(1)}_j = \sum_{i} W^{(1)}_{ji}\, x_i + b^{(1)}_j`,
    summary: "Each hidden unit multiplies every input by a weight, adds them up, and adds its bias.",
  },
  "hidden-act": {
    latex: String.raw`h_j = \tanh\!\left(z^{(1)}_j\right)`,
    summary: "A nonlinearity squashes each sum into (−1, 1). Without it, the whole network would collapse into one linear map and could not solve XOR.",
  },
  "output-sum": {
    latex: String.raw`z^{(2)}_k = \sum_{j} W^{(2)}_{kj}\, h_j + b^{(2)}_k`,
    summary: "The output layer repeats the weighted sum, this time over the hidden activations, producing one score (logit) per class.",
  },
  "softmax": {
    latex: String.raw`p_k = \frac{e^{z^{(2)}_k}}{\sum_m e^{z^{(2)}_m}} \qquad L = -\ln p_y`,
    summary: "Softmax turns the scores into probabilities. The loss is small when the true class gets high probability.",
  },
  "output-grad": {
    latex: String.raw`\delta^{(2)}_k = p_k - [k = y] \qquad \frac{\partial L}{\partial W^{(2)}_{kj}} = \delta^{(2)}_k\, h_j`,
    summary: "Backward pass begins. For softmax with cross-entropy, the error at each output is simply predicted minus target. Each output weight's gradient is that error times the activation it carried.",
  },
  "hidden-grad": {
    latex: String.raw`\delta^{(1)}_j = \Big(\sum_k W^{(2)}_{kj}\, \delta^{(2)}_k\Big)\,\big(1 - h_j^2\big) \qquad \frac{\partial L}{\partial W^{(1)}_{ji}} = \delta^{(1)}_j\, x_i`,
    summary: "The chain rule sends the output errors back along the same weights, scaled by the slope of tanh at each hidden unit.",
  },
  "update": {
    latex: String.raw`\theta \leftarrow \theta - \eta\, \frac{\partial L}{\partial \theta}`,
    summary: "Every weight and bias takes a small step against its gradient. Edge thickness and colour change to the new values.",
  },
};
