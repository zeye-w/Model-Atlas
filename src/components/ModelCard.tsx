import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import type { ModelDefinition } from "@/content/models";

const modeLabels: Record<ModelDefinition["executionMode"], string> = {
  "live-training": "Live training",
  "live-computation": "Live compute",
  "trace": "Trace replay",
  "planned": "Lab planned",
};

export function ModeBadge({ mode }: { mode: ModelDefinition["executionMode"] }) {
  const live = mode === "live-training" || mode === "live-computation";
  return <span className={`mode-badge ${live ? "mode-live" : "mode-planned"}`}>{modeLabels[mode]}</span>;
}

export function ModelCard({ model }: { model: ModelDefinition }) {
  return <Link className="model-card" href={`/models/${model.slug}`} style={{ "--model-color": model.color } as React.CSSProperties}>
    <div className="model-card-top"><span className="acronym">{model.acronym}</span><ModeBadge mode={model.executionMode}/></div>
    <h3>{model.name}</h3>
    <div className="model-card-translation">{model.chineseName}</div>
    <p>{model.summary}</p>
    <div className="model-card-bottom"><span>{model.family} · {model.level}</span><ArrowUpRight className="model-card-arrow" size={16}/></div>
  </Link>;
}
