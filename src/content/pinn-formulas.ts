/** Formulas shown on the PINN laboratory page (server-rendered with KaTeX). */
export const pinnFormulas = {
  ode: { label: "The equation", latex: String.raw`u'' + \mu\,u' + k\,u = 0,\quad u(0)=1,\; u'(0)=0` },
  data: { label: "Data loss", latex: String.raw`\mathcal{L}_{\text{data}} = \frac{1}{N}\sum_{i=1}^{N}\big(u_\theta(t_i) - y_i\big)^2` },
  physics: { label: "Physics loss", latex: String.raw`\mathcal{L}_{\text{phys}} = \frac{\lambda}{M}\sum_{j=1}^{M}\big(u_\theta'' + \mu\,u_\theta' + k\,u_\theta\big)^2\Big|_{t_j}` },
  ic: { label: "Initial condition", latex: String.raw`\mathcal{L}_{\text{IC}} = \lambda_{0}\Big[\big(u_\theta(0)-1\big)^2 + u_\theta'(0)^2\Big]` },
  derivative: { label: "Derivatives through a tanh layer", latex: String.raw`a = \tanh z,\quad a' = (1-a^2)\,z',\quad a'' = (1-a^2)\,z'' - 2a(1-a^2)\,z'^2` },
} as const;
export type PinnFormulaId = keyof typeof pinnFormulas;
