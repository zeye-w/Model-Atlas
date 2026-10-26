import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, FlaskConical } from "lucide-react";
import { ModeBadge } from "@/components/ModelCard";

export const metadata: Metadata = { title: "Laboratory" };

const roadmap = [
  { title:"K-Nearest Neighbors", detail:"Neighborhood selection, voting and dynamic classification boundaries." },
  { title:"Decision Tree", detail:"Watch impurity scores and recursive partitioning evolve." },
  { title:"Multilayer Perceptron", detail:"Inspect weight updates, activation functions and backpropagation." },
  { title:"CNN & Transformer", detail:"Explore exact tensor operations, feature maps and attention." },
];

export default function LabIndexPage() {
  return <div className="page-wrap"><div className="page-intro"><span className="section-index">03 / INTERACTIVE LABORATORY</span><h1>Ideas become experiments here.</h1><p>Every live lab exposes its execution mode, datasets, parameters and numerical results. We ship fully verifiable experiences incrementally.</p></div>
    <div className="lab-list"><div className="lab-list-card"><FlaskConical color="#6e9b90" size={27}/><ModeBadge mode="live-training"/><h2>Linear Regression</h2><p>Run gradient descent in your browser with seeded synthetic data. Track every update and export a reproducible record.</p><Link href="/lab/linear-regression" className="button-primary">Open the experiment <ArrowRight size={16}/></Link></div><div className="lab-list-card" style={{background:"#f2f6f5"}}><span className="section-index">COMING IN V1</span><h2>Next laboratories</h2>{roadmap.map((item)=><div key={item.title}><strong style={{fontSize:12}}>{item.title}</strong><p>{item.detail}</p></div>)}</div></div>
  </div>;
}
