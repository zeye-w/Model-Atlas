import type { ReactNode } from "react";
import { cnnLessons } from "@/content/cnn-theory";

/** `formulas` are server-rendered <Formula> nodes, one per lesson, so KaTeX never ships to the client. */
export function Theory({ open, formulas }: { open: (workspace: number) => void; formulas: ReactNode[] }) {
  return <div className="cnn-grid cnn-theory" lang="zh">
    {cnnLessons.map(({ title, symbols, explanation, example, workspace }, i) => <details className="cnn-card" key={title}>
      <summary lang="en">{title}</summary>
      {formulas[i]}
      <p>{symbols}</p><p>{explanation}</p><p>{example}</p>
      <button onClick={() => open(workspace)}>{workspace ? "打开训练台" : "打开卷积探索器"}</button>
    </details>)}
  </div>;
}
