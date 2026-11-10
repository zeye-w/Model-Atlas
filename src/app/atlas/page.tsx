import type { Metadata } from "next";
import Link from "next/link";
import { families, models } from "@/content/models";

export const metadata: Metadata = { title: "Algorithm Atlas" };

export default function AtlasPage() {
  return <div className="page-wrap"><div className="page-intro"><h1>See the bigger picture.</h1><p>Algorithms are not a single rigid family tree. This navigable overview groups models by their primary purpose; each model page also links to related approaches across families.</p></div>
    <div className="atlas-board"><div className="atlas-center"><strong>Machine learning</strong><span>Architectures · objectives · methods</span></div><div className="atlas-columns">{families.map((family)=><div className="atlas-family" key={family}><div className="atlas-family-title">{family}</div>{models.filter((model)=>model.family===family).map((model)=><Link key={model.slug} className="atlas-model-link" href={`/models/${model.slug}`} style={{"--model-color":model.color} as React.CSSProperties}><span className="acronym">{model.acronym}</span>{model.name}</Link>)}</div>)}</div><p className="atlas-footnote">Cross-family relationships are shown inside each model page. Categories are navigation aids: a CNN can also be part of a generative model, and PINNs commonly use an MLP backbone.</p></div>
  </div>;
}
