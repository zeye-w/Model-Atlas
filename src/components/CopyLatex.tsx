"use client";

import { Check, Copy } from "lucide-react";
import { useState } from "react";

export function CopyLatex({ latex }: { latex: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  async function copy() {
    try {
      await navigator.clipboard.writeText(latex);
      setState("copied");
    } catch {
      setState("failed");
    }
    window.setTimeout(() => setState("idle"), 1600);
  }
  return <button type="button" className="formula-copy" onClick={copy} title={latex} aria-label="Copy LaTeX source">
    {state === "copied" ? <Check size={13}/> : <Copy size={13}/>}
    <span aria-live="polite">{state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : "LaTeX"}</span>
  </button>;
}
