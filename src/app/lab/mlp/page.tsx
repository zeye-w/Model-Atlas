import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Formula } from "@/components/Formula";
import { ModeBadge } from "@/components/ModelCard";
import { MlpLab } from "@/components/MlpLab/MlpLab";
import { mlpPhaseText } from "@/content/mlp-phases";
import { phases } from "@/lib/mlp";
import "./mlp.css";

export const metadata: Metadata = {
  title: "MLP Laboratory",
  description: "Watch a forward pass, backpropagation and a gradient-descent update move through a small neural network, with every value computed in your browser.",
};

export default function MlpPage() {
  return <div className="page-wrap">
    <div className="detail-topline"><Link href="/lab" className="back-link"><ArrowLeft size={15}/> Laboratory</Link><ModeBadge mode="live-training"/></div>
    <div className="page-intro"><h1>Inside one training step.</h1><p>A two-input network with three hidden units learns the XOR pattern. Step through how a single example flows forward, how its error flows back, and how every weight moves.</p></div>
    <MlpLab
      phaseFormulas={phases.map((p) => <Formula key={p.id} className="mlp-formula" latex={mlpPhaseText[p.id].latex} label={p.title}/>)}
      phaseSummaries={phases.map((p) => mlpPhaseText[p.id].summary)}
    />
    <div className="quiet-note" style={{ marginTop: 16 }}><strong>What is real here.</strong> Every value, gradient and update is computed in your browser by the same code that the tests check against finite differences. Only the timing and the moving dashes are illustration.</div>
  </div>;
}
