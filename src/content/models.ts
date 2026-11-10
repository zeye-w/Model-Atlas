export type ModelFamily = "Classical ML" | "Ensemble Learning" | "Deep Learning" | "Generative AI" | "Scientific ML";
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
  },
  {
    slug: "adaboost", acronym: "ADA", name: "AdaBoost", chineseName: "自适应提升", family: "Ensemble Learning", paradigm: "Supervised Learning · Boosting", level: "Intermediate",
    summary: "Train weak learners in sequence, each focusing on the examples the last ones got wrong.",
    overview: "AdaBoost builds an ensemble one weak learner at a time and re-weights the training data after each round so that poorly predicted examples count more. For regression, AdaBoost.R2 scores every error relative to the worst one, turns the weighted average loss into a confidence for the new learner, and predicts with the weighted median of all learners.",
    architecture: ["Uniform sample weights", "Fit a shallow tree to weighted data", "Relative loss per example", "Learner confidence ln(1/β)", "Up-weight hard examples and repeat", "Weighted median of all learners"],
    latex: String.raw`\beta_m = \frac{\bar L_m}{1-\bar L_m}, \qquad w_i \leftarrow w_i\, \beta_m^{\,1-L_i}, \qquad \hat f(x) = \operatorname{median}_{\ln(1/\beta_m)}\big\{ h_m(x) \big\}`,
    formulaDescription: "AdaBoost.R2 (Drucker, 1997): L_i is example i's error divided by the largest error, L̄_m the weighted average. A learner with smaller L̄_m gets a larger say; boosting stops if L̄_m reaches 0.5.",
    strengths: ["Simple, few hyperparameters", "Concentrates capacity on hard regions", "Works with any weak learner"],
    limitations: ["Sensitive to noisy targets and outliers, which keep gaining weight", "Sequential, so training does not parallelise"],
    useCases: ["Tabular regression and classification", "Boosting shallow trees", "Teaching the idea of boosting"],
    tags: ["Boosting", "Ensemble", "Re-weighting"], executionMode: "live-training",
    paper: { name: "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting", url: "https://doi.org/10.1006/jcss.1997.1504", year: "1997" },
    related: ["gradient-boosting", "random-forest", "decision-tree"], color: "#a8724f",
    lab: { href: "/lab/regression?model=adaboost", title: "Regression laboratory", description: "Train AdaBoost, Gradient Boosting, Random Forest, CatBoost and an RBF network on the same data, replay how each fit is built, and compare their errors." }
  },
  {
    slug: "gradient-boosting", acronym: "GBM", name: "Gradient Boosting", chineseName: "梯度提升", family: "Ensemble Learning", paradigm: "Supervised Learning · Boosting", level: "Intermediate",
    summary: "Fit each new tree to the negative gradient of the loss and add it with a small step.",
    overview: "Gradient boosting performs gradient descent in function space. Starting from a constant, every round fits a regression tree to the negative gradient of the loss at the current predictions — for squared error, simply the residuals — and adds a shrunken copy of it. Small steps and shallow trees trade more rounds for better generalisation.",
    architecture: ["Constant initial prediction", "Compute residuals (negative gradient)", "Fit a regression tree to them", "Add the tree scaled by ν", "Repeat for M rounds"],
    latex: String.raw`r_i^{(m)} = -\left.\frac{\partial L(y_i, F)}{\partial F}\right|_{F=F_{m-1}(x_i)}, \qquad F_m(x) = F_{m-1}(x) + \nu\, h_m(x)`,
    formulaDescription: "Each tree h_m is fitted to the pseudo-residuals r. With squared loss these are y − F; the learning rate ν (shrinkage) controls how far each round moves.",
    strengths: ["State-of-the-art accuracy on tabular data", "Any differentiable loss", "Shrinkage and depth give fine control"],
    limitations: ["Many rounds and careful tuning", "Sequential training; can overfit without early stopping"],
    useCases: ["Tabular regression", "Ranking and risk scoring", "Strong baselines in competitions"],
    tags: ["Boosting", "Ensemble", "Gradient"], executionMode: "live-training",
    paper: { name: "Greedy Function Approximation: A Gradient Boosting Machine", url: "https://doi.org/10.1214/aos/1013203451", year: "2001" },
    related: ["catboost", "adaboost", "decision-tree"], color: "#b9673f",
    lab: { href: "/lab/regression?model=gradient-boosting", title: "Regression laboratory", description: "Train AdaBoost, Gradient Boosting, Random Forest, CatBoost and an RBF network on the same data, replay how each fit is built, and compare their errors." }
  },
  {
    slug: "random-forest", acronym: "RF", name: "Random Forest", chineseName: "随机森林", family: "Ensemble Learning", paradigm: "Supervised Learning · Bagging", level: "Foundation",
    summary: "Average many deep trees, each grown on a bootstrap sample with random feature choices.",
    overview: "A random forest grows many decision trees independently, each on a bootstrap resample of the training set and, at every split, on a random subset of features. Individual trees overfit; averaging them cancels much of that variance while keeping their flexibility.",
    architecture: ["Draw a bootstrap sample", "Grow a deep tree (random feature subset per split)", "Repeat for B trees", "Average the trees' predictions"],
    latex: String.raw`\hat f(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x), \qquad T_b \text{ trained on a bootstrap sample}`,
    formulaDescription: "Averaging B decorrelated trees reduces variance; adding trees does not cause overfitting, it only stabilises the average.",
    strengths: ["Strong default with little tuning", "Parallel training", "Built-in out-of-bag error estimate"],
    limitations: ["Piecewise-constant, cannot extrapolate", "Large forests are slow and memory hungry"],
    useCases: ["Tabular regression and classification", "Feature importance screening", "Robust baselines"],
    tags: ["Bagging", "Ensemble", "Variance Reduction"], executionMode: "live-training",
    paper: { name: "Random Forests", url: "https://doi.org/10.1023/A:1010933404324", year: "2001" },
    related: ["decision-tree", "gradient-boosting", "adaboost"], color: "#6f8a63",
    lab: { href: "/lab/regression?model=random-forest", title: "Regression laboratory", description: "Train AdaBoost, Gradient Boosting, Random Forest, CatBoost and an RBF network on the same data, replay how each fit is built, and compare their errors." }
  },
  {
    slug: "catboost", acronym: "CAT", name: "CatBoost", chineseName: "CatBoost 梯度提升", family: "Ensemble Learning", paradigm: "Supervised Learning · Boosting", level: "Advanced",
    summary: "Gradient boosting with symmetric trees, regularised leaves and unbiased handling of categorical features.",
    overview: "CatBoost is a gradient boosting library whose default trees are symmetric (oblivious): every node on a level uses the same split, giving fast, regular models. Leaf values are shrunk by an L2 term. Its paper's key ideas — ordered boosting and ordered target statistics — remove the prediction shift that arises when the same examples are used both to compute gradients and to fit trees, and encode categorical features without target leakage.",
    architecture: ["Constant initial prediction", "Residuals of current model", "Symmetric tree: one split per level", "L2-regularised leaf values", "Add with learning rate; repeat"],
    latex: String.raw`F_m = F_{m-1} + \eta\, h_m, \qquad h_m(x) = w_{\ell(x)}, \quad w_\ell = \frac{\sum_{i \in \ell} r_i}{|\ell| + \lambda}`,
    formulaDescription: "Each symmetric tree assigns x to leaf ℓ(x) by the same condition on every level. The leaf value is the sum of residuals divided by count plus l2_leaf_reg λ (default 3).",
    strengths: ["Strong results with default settings", "Native categorical features without leakage", "Fast inference thanks to symmetric trees"],
    limitations: ["Symmetric trees are less flexible per tree", "Ordered mode is slower to train"],
    useCases: ["Tabular data with many categorical columns", "Click-through and recommendation", "Production gradient boosting"],
    tags: ["Boosting", "Symmetric Trees", "Categorical"], executionMode: "live-training",
    paper: { name: "CatBoost: unbiased boosting with categorical features", url: "https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html", year: "2018" },
    related: ["gradient-boosting", "random-forest", "decision-tree"], color: "#c08a3e",
    lab: { href: "/lab/regression?model=catboost", title: "Regression laboratory", description: "Train AdaBoost, Gradient Boosting, Random Forest, CatBoost and an RBF network on the same data, replay how each fit is built, and compare their errors." }
  },
  {
    slug: "rbf-network", acronym: "RBF", name: "RBF Network", chineseName: "径向基函数网络", family: "Deep Learning", paradigm: "Supervised Learning · Kernel", level: "Intermediate",
    summary: "One hidden layer of Gaussian bumps, combined linearly to approximate a smooth function.",
    overview: "A radial basis function network has a single hidden layer whose units respond to the distance between the input and a centre, usually with a Gaussian. With centres and width fixed, the output layer is linear, so its weights have a closed-form least-squares solution. It sits between kernel methods and neural networks.",
    architecture: ["Input x", "Distance to each centre c_k", "Gaussian activation φ_k(x)", "Weighted sum plus bias", "Weights by (ridge) least squares"],
    latex: String.raw`\hat f(x) = b + \sum_{k=1}^{K} w_k \exp\!\left(-\frac{\lVert x - c_k\rVert^2}{2\sigma^2}\right), \qquad \mathbf{w} = (\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top \mathbf{y}`,
    formulaDescription: "Φ holds every training input's activation on every unit. A small ridge term λ keeps the solve stable when units overlap.",
    strengths: ["Smooth fits with few units", "Fast, exact training of the output layer", "Interpretable local units"],
    limitations: ["Centres and width must be chosen well", "Number of units grows quickly with input dimension"],
    useCases: ["Function approximation", "Time-series prediction", "Surrogate models in engineering"],
    tags: ["Neural Network", "Kernel", "Least Squares"], executionMode: "live-training",
    paper: { name: "Multivariable Functional Interpolation and Adaptive Networks", url: "https://wpmedia.wolfram.com/sites/13/2018/02/02-3-5.pdf", year: "1988" },
    related: ["mlp", "gradient-boosting", "linear-regression"], color: "#5f7a96",
    lab: { href: "/lab/regression?model=rbf-network", title: "Regression laboratory", description: "Train AdaBoost, Gradient Boosting, Random Forest, CatBoost and an RBF network on the same data, replay how each fit is built, and compare their errors." }
  }
];

export const families: ModelFamily[] = ["Classical ML", "Ensemble Learning", "Deep Learning", "Generative AI", "Scientific ML"];

export const familyDescriptions: Record<ModelFamily, string> = {
  "Classical ML": "The foundations of inference, decisions and optimization.",
  "Ensemble Learning": "Combine many simple models — by voting, averaging or boosting — into a strong one.",
  "Deep Learning": "From layers and convolution to contextual attention.",
  "Generative AI": "Understand the processes behind synthetic data.",
  "Scientific ML": "Bring equations and physical constraints into learning.",
};

export const getModel = (slug: string) => models.find((model) => model.slug === slug);

export const isLiveMode = (mode: ExecutionMode) => mode === "live-training" || mode === "live-computation";

/** Models with a shipped, executable laboratory. */
export const liveLabs = models.filter((model) => model.lab && isLiveMode(model.executionMode));
/** The route a lab link points to, without query or hash. */
export const labPath = (href: string) => href.split(/[?#]/)[0];
/** One entry per laboratory page (several models can share one lab). */
export const labPages = liveLabs.filter((model, i) => liveLabs.findIndex((m) => labPath(m.lab!.href) === labPath(model.lab!.href)) === i)
  .map((model) => ({ ...model.lab!, href: labPath(model.lab!.href), mode: model.executionMode, models: liveLabs.filter((m) => labPath(m.lab!.href) === labPath(model.lab!.href)) }));
/** Models whose laboratory is on the roadmap. */
export const plannedLabs = models.filter((model) => model.labPlan && !model.lab);
export const featuredModels = models.filter((model) => model.featured);
export const modelsInFamily = (family: ModelFamily) => models.filter((model) => model.family === family);

/** "8" → "08", matching the editorial numbering used across the site. */
export const pad2 = (value: number) => String(value).padStart(2, "0");
export const countLabel = (value: number, noun: string) => `${pad2(value)} ${noun}${value === 1 ? "" : "s"}`;
