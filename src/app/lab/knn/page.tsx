import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { KnnLab } from "@/components/ClassifyLab/KnnLab";
import { ModeBadge } from "@/components/ModelCard";
import "../steps.css";
import "../regression/regression.css";
import "../classify.css";

export const metadata: Metadata = {
  title: "KNN Laboratory",
  description: "Drag a query point and watch k-nearest neighbours measure distances, pick the closest points and vote — and see why feature scaling and the choice of k matter.",
};

export default function KnnPage() {
  return <div className="page-wrap">
    <div className="detail-topline"><Link href="/lab" className="back-link"><ArrowLeft size={15}/> Laboratory</Link><ModeBadge mode="live-computation"/></div>
    <div className="page-intro"><h1>Ask the neighbours.</h1><p>K-nearest neighbours keeps every training point and decides by asking the closest ones. Move the query, change k and the distance, and see what happens when one feature is measured in different units.</p></div>
    <KnnLab/>
    <div className="quiet-note" style={{ marginTop: 16 }}><strong>What is real here.</strong> Every distance, vote and coloured region is computed in your browser. Tests check the distance formulas, sorting, tie-breaking, distance weighting, that k = 1 memorises the training set, and that an unscaled feature lowers accuracy until the features are standardised.</div>
  </div>;
}
