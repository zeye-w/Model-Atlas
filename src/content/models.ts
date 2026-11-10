export type ModelFamily = "Classical ML" | "Deep Learning" | "Generative AI" | "Scientific ML";
export type ExecutionMode = "live-training" | "live-computation" | "trace" | "planned";

export interface ModelDefinition {
  slug: string;
  acronym: string;
  name: string;
  chineseName: string;
  family: ModelFamily;
  paradigm: string;
  level: "Foundation" | "Intermediate" | "Advanced";
  summary: string;
  overview: string;
  architecture: string[];
  /** KaTeX-compatible LaTeX, rendered on the server. */
  latex: string;
  formulaDescription: string;
  strengths: string[];
  limitations: string[];
  useCases: string[];
  tags: string[];
  executionMode: ExecutionMode;
  paper: { name: string; url: string; year: string };
  related: string[];
  color: string;
  /** Shown in "Selected models" on the home page. */
  featured?: boolean;
  /** A shipped laboratory. Required when executionMode is not "planned". */
  lab?: { href: string; title: string; description: string };
  /** Roadmap line for a laboratory that has not shipped yet. */
  labPlan?: string;
}

export const models: ModelDefinition[] = [
  {
    slug: "linear-regression", acronym: "LR", name: "Linear Regression", chineseName: "线性回归", family: "Classical ML", paradigm: "Supervised Learning", level: "Foundation",
    summary: "The simplest lens into prediction, optimization, and generalization.",
    overview: "Linear regression fits a weighted sum of input features to a continuous target. It is a starting point for understanding model parameters, residuals, gradient descent, and the difference between fitting and generalizing.",
    architecture: ["Input features x", "Weighted sum wx + b", "Predicted target ŷ", "Squared-error objective", "Gradient-based parameter update"],
    latex: String.raw`\hat{y} = wx + b \qquad \mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} \left(\hat{y}_i - y_i\right)^2`,
    formulaDescription: "The parameters w and b minimize mean squared prediction error. The lab computes the exact gradients over a synthetic dataset.",
    strengths: ["Fast, interpretable baseline", "Clear mathematical assumptions", "Effective for approximately linear relationships"],
    limitations: ["Needs feature engineering for complex nonlinear structure", "Sensitive to outliers without robust losses"],
    useCases: ["Regression baseline", "Trend estimation", "Interpretable forecasting"],
    tags: ["Regression", "Optimization", "Interpretable"], executionMode: "live-training",
    paper: { name: "Least squares — historical overview", url: "https://mathworld.wolfram.com/LeastSquaresFitting.html", year: "Classical" },
    related: ["mlp", "knn"], color: "#b96a4b", featured: true,
    lab: { href: "/lab/linear-regression", title: "Linear Regression", description: "Run gradient descent in your browser with seeded synthetic data. Track every update and export a reproducible record." }
  },
  {
    slug: "knn", acronym: "KNN", name: "K-Nearest Neighbors", chineseName: "K 近邻", family: "Classical ML", paradigm: "Supervised Learning", level: "Foundation",
    summary: "Predict from nearby examples rather than learning a global equation.",
    overview: "KNN predicts from nearby labeled samples under a selected distance metric. It makes local structure, feature scaling and the bias–variance trade-off visible.",
    architecture: ["Store labeled examples", "Choose a query point", "Compute distances", "Select K nearest samples", "Aggregate labels or targets"],
    latex: String.raw`d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j} \left(x_j - x_{ij}\right)^2}`,
    formulaDescription: "Euclidean distance is one common metric. For classification, the K closest examples can vote for a class.",
    strengths: ["Intuitive local predictions", "Non-parametric decisions", "No explicit fitting stage"],
    limitations: ["Sensitive to scaling and irrelevant features", "Query cost grows with training set size"],
    useCases: ["Classification", "Similarity search", "Simple regression"],
    tags: ["Classification", "Distance", "Non-parametric"], executionMode: "planned",
    paper: { name: "Discriminatory Analysis: Nonparametric Discrimination", url: "https://doi.org/10.2172/4211277", year: "1951" },
    related: ["decision-tree", "linear-regression"], color: "#7f8a73", labPlan: "Neighborhood selection, voting and dynamic classification boundaries."
  },
  {
    slug: "decision-tree", acronym: "DT", name: "Decision Tree", chineseName: "决策树", family: "Classical ML", paradigm: "Supervised Learning", level: "Foundation",
    summary: "Turn feature thresholds into a transparent sequence of decisions.",
    overview: "A decision tree partitions input space with feature-based splits. Its structure reveals the trade-off between interpretability and overfitting.",
    architecture: ["Collect labeled samples", "Score candidate feature splits", "Choose the best split", "Repeat for child nodes", "Predict at a leaf"],
    latex: String.raw`\operatorname{Gini} = 1 - \sum_{k} p_k^2`,
    formulaDescription: "Gini impurity is one possible criterion for classification splits. Lower values indicate purer child nodes.",
    strengths: ["Rule-based explanations", "Captures nonlinear interactions", "Little feature preprocessing needed"],
    limitations: ["Deep trees can overfit", "Small data changes may change the tree"],
    useCases: ["Tabular classification", "Feature interactions", "Decision support"],
    tags: ["Classification", "Tree", "Interpretable"], executionMode: "planned",
    paper: { name: "Classification and Regression Trees", url: "https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Olshen-Stone/p/book/9780412048418", year: "1984" },
    related: ["knn", "mlp"], color: "#b08a58", labPlan: "Watch impurity scores and recursive partitioning evolve."
  },
  {
    slug: "mlp", acronym: "MLP", name: "Multilayer Perceptron", chineseName: "多层感知机", family: "Deep Learning", paradigm: "Supervised Learning", level: "Intermediate",
    summary: "Stack learned nonlinear transformations to approximate complex functions.",
    overview: "An MLP uses fully connected layers, nonlinear activations, and backpropagation. It bridges basic regression and modern deep learning.",
    architecture: ["Input layer", "Dense linear projection", "Nonlinear activation", "Additional hidden layers", "Output and backpropagation"],
    latex: String.raw`\mathbf{h}^{(l)} = \sigma\left(W^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)`,
    formulaDescription: "Each layer transforms the preceding representation, with gradients flowing backward via the chain rule.",
    strengths: ["Flexible nonlinear approximation", "Foundation for deep learning", "General-purpose feature modeling"],
    limitations: ["Requires tuning and data", "Does not directly exploit image or graph structure"],
    useCases: ["Regression", "Classification", "Neural network fundamentals"],
    tags: ["Neural Network", "Backpropagation", "Optimization"], executionMode: "live-training",
    paper: { name: "Learning representations by back-propagating errors", url: "https://doi.org/10.1038/323533a0", year: "1986" },
    related: ["linear-regression", "cnn", "transformer"], color: "#988373", lab: { href: "/lab/mlp", title: "MLP laboratory", description: "Watch one training step move through a small network: weighted sums, activations, softmax, backpropagated errors and the weight update — every number computed live." }
  },
  {
    slug: "cnn", acronym: "CNN", name: "Convolutional Neural Network", chineseName: "卷积神经网络", family: "Deep Learning", paradigm: "Supervised / Self-supervised", level: "Intermediate",
    summary: "Learn local patterns through shared filters and hierarchical feature maps.",
    overview: "CNNs apply learned kernels across spatially organized signals. Shared weights and receptive fields help discover edges, textures and larger structures across layers.",
    architecture: ["Image or spatial input", "Learned convolution filters", "Feature maps and activation", "Pooling or downsampling", "Prediction head"],
    latex: String.raw`Y[i, j] = \sum_{m}\sum_{n} K[m, n]\, X[i + m, j + n]`,
    formulaDescription: "The common deep-learning implementation is cross-correlation (no kernel flip), although usually called convolution.",
    strengths: ["Strong spatial inductive bias", "Shared parameters", "Useful for images and seismic volumes"],
    limitations: ["Global context may need deep stacks", "Training can be compute intensive"],
    useCases: ["Computer vision", "Seismic interpretation", "Image segmentation"],
    tags: ["Vision", "Convolution", "Feature Maps"], executionMode: "live-training",
    paper: { name: "Gradient-Based Learning Applied to Document Recognition", url: "https://doi.org/10.1109/5.726791", year: "1998" },
    related: ["mlp", "transformer", "pinn"], color: "#bd704f", featured: true,
    lab: { href: "/lab/cnn", title: "CNN laboratory", description: "Inspect convolution windows, train a waveform classifier, and trace weight updates." }
  },
  {
    slug: "transformer", acronym: "TRF", name: "Transformer", chineseName: "Transformer 架构", family: "Deep Learning", paradigm: "Self-supervised / Supervised", level: "Advanced",
    summary: "Use attention to connect positions across a sequence.",
    overview: "The Transformer builds contextual representations with attention and feed-forward blocks. Attention reveals input-dependent interactions but should not be mistaken for a complete model explanation.",
    architecture: ["Input token embeddings", "Query / Key / Value projections", "Scaled dot-product attention", "Feed-forward transformation", "Residual paths and normalization"],
    latex: String.raw`\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V`,
    formulaDescription: "Queries and keys produce attention scores; softmax yields weights that combine the values.",
    strengths: ["Long-range relationships", "Flexible across modalities", "Parallelizable training"],
    limitations: ["Standard attention may scale quadratically with length", "Attention maps are not causal explanations"],
    useCases: ["Language models", "Vision transformers", "Cross-modal fusion"],
    tags: ["Attention", "Sequence", "Foundation Models"], executionMode: "planned",
    paper: { name: "Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762", year: "2017" },
    related: ["cnn", "mlp", "diffusion"], color: "#87796e", featured: true, labPlan: "Explore attention and token interactions."
  },
  {
    slug: "diffusion", acronym: "DIF", name: "Diffusion Model", chineseName: "扩散模型", family: "Generative AI", paradigm: "Generative / Self-supervised", level: "Advanced",
    summary: "Generate structured samples by learning to reverse a noise process.",
    overview: "Denoising diffusion models learn a reverse process that transforms noise toward samples drawn from the data distribution. Honest trajectory replay can reveal sampling stages without claiming live training.",
    architecture: ["Data sample", "Forward noising schedule", "Noise-conditioned denoiser", "Reverse sampling steps", "Generated sample"],
    latex: String.raw`\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\, \boldsymbol{\epsilon}`,
    formulaDescription: "Clean data mixes with Gaussian noise at step t; a learned model approximates the reverse process.",
    strengths: ["Generative flexibility", "Conditional generation", "Scientific prior modeling"],
    limitations: ["Sampling can be expensive", "Samples require careful evaluation"],
    useCases: ["Image generation", "Data augmentation", "Generative priors"],
    tags: ["Generative", "Denoising", "Sampling"], executionMode: "planned",
    paper: { name: "Denoising Diffusion Probabilistic Models", url: "https://arxiv.org/abs/2006.11239", year: "2020" },
    related: ["transformer", "pinn"], color: "#bd8577"
  },
  {
    slug: "pinn", acronym: "PINN", name: "Physics-Informed Neural Network", chineseName: "物理信息神经网络", family: "Scientific ML", paradigm: "Physics-guided / Supervised", level: "Advanced",
    summary: "Couple data fitting with differential-equation constraints.",
    overview: "PINNs train a neural approximation while penalizing violations of governing equations, boundary conditions and observations. They are relevant for scientific inverse problems.",
    architecture: ["Coordinates and parameters", "Neural field approximation", "Automatic differentiation", "Equation and boundary residuals", "Composite training objective"],
    latex: String.raw`\mathcal{L} = \lambda_{\text{data}} \mathcal{L}_{\text{data}} + \lambda_{\text{PDE}} \mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}}`,
    formulaDescription: "Physics, observations and boundary conditions contribute weighted losses; balancing depends on the problem and scaling.",
    strengths: ["Uses governing equations", "Can integrate sparse observations", "Relevant to inverse problems"],
    limitations: ["Loss balancing is difficult", "Can struggle with multiscale dynamics"],
    useCases: ["Wave equations", "Flow modeling", "Scientific inversion"],
    tags: ["Physics", "PDE", "Inverse Problems"], executionMode: "planned",
    paper: { name: "Physics-informed neural networks: A deep learning framework", url: "https://doi.org/10.1016/j.jcp.2018.10.045", year: "2019" },
    related: ["mlp", "cnn", "diffusion"], color: "#7f8a73"
  }
];

export const families: ModelFamily[] = ["Classical ML", "Deep Learning", "Generative AI", "Scientific ML"];

export const familyDescriptions: Record<ModelFamily, string> = {
  "Classical ML": "The foundations of inference, decisions and optimization.",
  "Deep Learning": "From layers and convolution to contextual attention.",
  "Generative AI": "Understand the processes behind synthetic data.",
  "Scientific ML": "Bring equations and physical constraints into learning.",
};

export const getModel = (slug: string) => models.find((model) => model.slug === slug);

export const isLiveMode = (mode: ExecutionMode) => mode === "live-training" || mode === "live-computation";

/** Models with a shipped, executable laboratory. */
export const liveLabs = models.filter((model) => model.lab && isLiveMode(model.executionMode));
/** Models whose laboratory is on the roadmap. */
export const plannedLabs = models.filter((model) => model.labPlan && !model.lab);
export const featuredModels = models.filter((model) => model.featured);
export const modelsInFamily = (family: ModelFamily) => models.filter((model) => model.family === family);

/** "8" → "08", matching the editorial numbering used across the site. */
export const pad2 = (value: number) => String(value).padStart(2, "0");
export const countLabel = (value: number, noun: string) => `${pad2(value)} ${noun}${value === 1 ? "" : "s"}`;
