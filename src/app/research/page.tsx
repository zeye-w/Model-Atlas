import type { Metadata } from "next";
import { ExternalLink } from "lucide-react";
import { models } from "@/content/models";

export const metadata: Metadata = { title: "Research References" };

export default function ResearchPage() {
  return <div className="page-wrap"><div className="page-intro"><h1>Follow the source.</h1><p>Browse original and foundational references associated with each curated model. These links provide a starting point for deeper reading, not an exhaustive bibliography.</p></div><div className="resource-grid">{models.map((model)=><div className="resource-card" key={model.slug} style={{"--model-color":model.color} as React.CSSProperties}><div className="resource-meta"><span className="acronym">{model.acronym}</span>{model.paper.year}</div><h3>{model.paper.name}</h3><p>{model.name} · {model.family}</p><a href={model.paper.url} target="_blank" rel="noreferrer">Open reference <ExternalLink size={14}/></a></div>)}</div></div>;
}
