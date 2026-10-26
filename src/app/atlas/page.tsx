import type { Metadata } from "next";
import Link from "next/link";
import { families, models } from "@/content/models";

export const metadata: Metadata = { title: "Algorithm Atlas" };

export default function AtlasPage() {
  return <div className="page-wrap"><div className="page-intro"><span className="section-index">02 / ALGORITHM ATLAS</span><h1>See the bigger picture.</h1><p>Algorithms are not a single rigid family tree. This navigable overview groups models by their primary purpose; each model page also links to related approaches across families.</p></div>
    <div className="atlas-board"><div className="atlas-label">NAVIGABLE MODEL MAP / VERSION 0.1</div><div className="atlas-center">MACHINE LEARNING<br/><small style={{fontSize:10,fontWeight:500,color:"#82a0a8"}}>Architectures · Objectives · Methods</small></div><div className="atlas-columns">{families.map((family)=><div className="atlas-family" key={family}><div className="atlas-family-title">{family}</div>{models.filter((model)=>model.family===family).map((model)=><Link key={model.slug} className="atlas-model-link" href={`/models/${model.slug}`}><span>{model.acronym}</span>{model.name}</Link>)}</div>)}</div><p className="atlas-footnote">Cross-family relationships are shown inside each model page. Categories are navigation aids: a CNN can also be part of a generative model, and PINNs commonly use an MLP backbone.</p></div>
  </div>;
}
