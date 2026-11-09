import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, FlaskConical } from "lucide-react";
import { ModeBadge } from "@/components/ModelCard";
import { liveLabs, plannedLabs } from "@/content/models";

export const metadata: Metadata = { title: "Laboratory" };

export default function LabIndexPage() {
  return <div className="page-wrap"><div className="page-intro"><span className="section-index">03 / INTERACTIVE LABORATORY</span><h1>Ideas become experiments here.</h1><p>Every live lab exposes its execution mode, datasets, parameters and numerical results. We ship fully verifiable experiences incrementally.</p></div>
    <div className="lab-list">
      {liveLabs.map((model) => <div className="lab-list-card" key={model.slug}><FlaskConical color={model.color} size={27}/><ModeBadge mode={model.executionMode}/><h2>{model.lab!.title}</h2><p>{model.lab!.description}</p><Link href={model.lab!.href} className="button-primary">Open {model.lab!.title} <ArrowRight size={16}/></Link></div>)}
      {plannedLabs.length > 0 && <div className="lab-list-card" style={{background:"#f2f6f5"}}><span className="section-index">COMING IN V1</span><h2>Next laboratories</h2>{plannedLabs.map((model)=><div key={model.slug}><strong style={{fontSize:12}}>{model.name}</strong><p>{model.labPlan}</p></div>)}</div>}
    </div>
  </div>;
}
