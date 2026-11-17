import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Formula } from "@/components/Formula";
import { ModeBadge } from "@/components/ModelCard";
import { PinnLab } from "@/components/PinnLab/PinnLab";
import { pinnFormulas, type PinnFormulaId } from "@/content/pinn-formulas";
import "../steps.css";
import "../regression/regression.css";
import "./pinn.css";

export const metadata: Metadata = {
  title: "PINN Laboratory",
  description: "Train two neural networks on a damped oscillator in your browser — one on data alone, one with the physics residual — and see which can predict beyond the data and recover an unknown damping coefficient.",
};

export default function PinnPage() {
  const formulas = Object.fromEntries(Object.entries(pinnFormulas).map(([id, f]) => [id, <Formula key={id} className="mlp-formula" latex={f.latex} label={f.label}/>])) as Record<PinnFormulaId, React.ReactNode>;
  return <div className="page-wrap">
    <div className="detail-topline"><Link href="/lab" className="back-link"><ArrowLeft size={15}/> Laboratory</Link><ModeBadge mode="live-training"/></div>
    <div className="page-intro"><h1>Teach a network the equation, not just the data.</h1><p>A damped spring, a handful of noisy measurements and two identical networks. One learns from the measurements alone; the other is also penalised wherever it breaks the differential equation. Train both and compare what they predict where nobody measured.</p></div>
    <PinnLab formulas={formulas}/>
    <div className="quiet-note" style={{ marginTop: 16 }}><strong>What is real here.</strong> Both networks train in your browser with Adam, from the same seeded initial weights. The derivatives u′ and u″ are exact (propagated through every layer, not finite differences) and their parameter gradients are written out by hand; tests check both against finite differences, check that the PINN beats the data-only fit beyond the observations, and that it recovers μ to within 0.6. Real PINN work uses automatic differentiation in a framework, partial differential equations in several dimensions, and usually adds an L-BFGS phase and careful loss weighting — the λ setting shows why.</div>
  </div>;
}
