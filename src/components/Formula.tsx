import "katex/dist/katex.min.css";
import { renderLatex } from "@/lib/math";
import { CopyLatex } from "./CopyLatex";

/**
 * Server component: renders a formula with KaTeX at build/request time.
 * The MathML half of KaTeX's output is what screen readers announce.
 */
export function Formula({ latex, className, label }: { latex: string; className?: string; label?: string }) {
  return <figure className={`formula ${className ?? ""}`} aria-label={label}>
    <div className="formula-render" dangerouslySetInnerHTML={{ __html: renderLatex(latex) }}/>
    <CopyLatex latex={latex}/>
  </figure>;
}
