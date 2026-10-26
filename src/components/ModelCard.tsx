import Link from "next/link";
import { ArrowUpRight, MoveUpRight } from "lucide-react";
import type { ModelDefinition } from "@/content/models";

export function ModeBadge({ mode }: { mode: ModelDefinition["executionMode"] }) {
  const label = mode === "live-training" ? "LIVE TRAINING" : mode === "live-computation" ? "LIVE COMPUTE" : mode === "trace" ? "TRACE REPLAY" : "LAB PLANNED";
  return <span className={`mode-badge ${mode === "live-training" || mode === "live-computation" ? "mode-live" : "mode-planned"}`}>{label}</span>;
}

export function ModelCard({ model, number }: { model: ModelDefinition; number?: number }) {
  return <Link className="model-card" href={`/models/${model.slug}`}>
    <div className="model-card-top"><span className="model-card-number">{String(number ?? 1).padStart(2, "0")} / MODEL</span><ArrowUpRight size={18}/></div>
    <div className="model-symbol" style={{ "--model-color": model.color } as React.CSSProperties}>
      <span>{model.acronym}</span><i/><i/><i/>
    </div>
    <div className="model-card-body"><div className="model-card-family">{model.family} <MoveUpRight size={11}/></div><h3>{model.name}</h3><div className="model-card-translation">{model.chineseName}</div><p>{model.summary}</p></div>
    <div className="model-card-bottom"><ModeBadge mode={model.executionMode}/><span>{model.level}</span></div>
  </Link>;
}
