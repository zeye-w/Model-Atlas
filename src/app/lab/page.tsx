import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, FlaskConical } from "lucide-react";
import { ModeBadge } from "@/components/ModelCard";
import { labPages, plannedLabs } from "@/content/models";

export const metadata: Metadata = { title: "Laboratory" };

export default function LabIndexPage() {
  return <div className="page-wrap"><div className="page-intro"><h1>Ideas become experiments here.</h1><p>Every live lab exposes its execution mode, datasets, parameters and numerical results. We ship fully verifiable experiences incrementally.</p></div>
    <div className="lab-list">
      {labPages.map((lab) => <div className="lab-list-card" key={lab.href}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",width:"100%"}}><span className="lab-icon"><FlaskConical size={19} strokeWidth={1.7}/></span><ModeBadge mode={lab.mode}/></div>
        <h2>{lab.title}</h2><p>{lab.description}</p>
        {lab.models.length > 1 && <div className="tag-row">{lab.models.map((m) => <span className="tag" key={m.slug}>{m.name}</span>)}</div>}
        <Link href={lab.href} className="button-primary">Open {lab.title} <ArrowRight size={16}/></Link>
      </div>)}
      {plannedLabs.length > 0 && <div className="lab-list-card lab-roadmap"><h2>Coming next</h2>
        <ul>{plannedLabs.map((model)=><li key={model.slug}><strong>{model.name}</strong><span>{model.labPlan}</span></li>)}</ul>
      </div>}
    </div>
  </div>;
}
