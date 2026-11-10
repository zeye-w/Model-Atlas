import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, FlaskConical } from "lucide-react";
import { ModeBadge } from "@/components/ModelCard";
import { liveLabs, plannedLabs } from "@/content/models";

export const metadata: Metadata = { title: "Laboratory" };

export default function LabIndexPage() {
  return <div className="page-wrap"><div className="page-intro"><h1>Ideas become experiments here.</h1><p>Every live lab exposes its execution mode, datasets, parameters and numerical results. We ship fully verifiable experiences incrementally.</p></div>
    <div className="lab-list">
      {liveLabs.map((model) => <div className="lab-list-card" key={model.slug}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",width:"100%"}}><span className="lab-icon"><FlaskConical size={19} strokeWidth={1.7}/></span><ModeBadge mode={model.executionMode}/></div>
        <h2>{model.lab!.title}</h2><p>{model.lab!.description}</p>
        <Link href={model.lab!.href} className="button-primary">Open {model.lab!.title} <ArrowRight size={16}/></Link>
      </div>)}
      {plannedLabs.length > 0 && <div className="lab-list-card lab-roadmap"><h2>Coming next</h2>
        <ul>{plannedLabs.map((model)=><li key={model.slug}><strong>{model.name}</strong><span>{model.labPlan}</span></li>)}</ul>
      </div>}
    </div>
  </div>;
}
