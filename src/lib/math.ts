import katex from "katex";

/**
 * Render LaTeX to HTML + MathML. Runs on the server (and in Node tests), so
 * pages ship the rendered markup and KaTeX's CSS, not its JavaScript.
 * `throwOnError` makes a malformed formula fail the build instead of shipping
 * red error text.
 */
export function renderLatex(latex: string, displayMode = true): string {
  return katex.renderToString(latex, {
    displayMode,
    throwOnError: true,
    output: "htmlAndMathml",
    strict: "error",
    trust: false,
  });
}
