import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { DecisionTreeLab } from "@/components/ClassifyLab/DecisionTreeLab";
import { ModeBadge } from "@/components/ModelCard";
import "../steps.css";
import "../regression/regression.css";
import "../classify.css";

export const metadata: Metadata = {
  title: "Decision Tree Laboratory",
  description: "Grow a CART classification tree one split at a time: see every threshold it tries, the impurity decrease it maximises, and how depth trades training fit for held-out accuracy.",
};

export default function DecisionTreePage() {
  return <div className="page-wrap">
    <div className="detail-topline"><Link href="/lab" className="back-link"><ArrowLeft size={15}/> Laboratory</Link><ModeBadge mode="live-training"/></div>
    <div className="page-intro"><h1>A tree, one question at a time.</h1><p>Watch a classification tree carve the plane into rectangles. Each split is the single threshold, on one feature, that most reduces impurity — chosen greedily, one node at a time.</p></div>
    <DecisionTreeLab/>
    <div className="quiet-note" style={{ marginTop: 16 }}><strong>What is real here.</strong> The tree is grown in your browser by this site’s CART implementation (Gini or entropy, exhaustive threshold search, breadth-first growth so it can be replayed). Tests check the impurity formulas, that each chosen split matches a brute-force search, and that children partition their parent.</div>
  </div>;
}
