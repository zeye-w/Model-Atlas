import type { Metadata } from "next";
import { ExternalLink } from "lucide-react";
import { models } from "@/content/models";

export const metadata: Metadata = { title: "Research References" };

export default function ResearchPage() {
  return <div className="page-wrap"><div className="page-intro"><span className="section-index">05 / RESEARCH</span><h1>Follow the source.</h1><p>Browse original and foundational references associated with each curated model. These links provide a starting point for deeper reading, not an exhaustive bibliography.</p></div><div className="resource-grid">{models.map((model)=><div className="resource-card" key={model.slug}><span className="section-index">{model.acronym} / {model.paper.year}</span><h3 style={{marginTop:15}}>{model.paper.name}</h3><p>{model.name} · {model.family}</p><a href={model.paper.url} target="_blank" rel="noreferrer">Open reference <ExternalLink size={14}/></a></div>)}</div></div>;
}
