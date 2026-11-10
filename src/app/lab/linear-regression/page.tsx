import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { RegressionLab } from "@/components/RegressionLab";
import { ModeBadge } from "@/components/ModelCard";

export const metadata: Metadata = { title: "Linear Regression Lab" };

export default function LinearRegressionLabPage() {
  return <div className="page-wrap"><div className="detail-topline"><Link href="/lab" className="back-link"><ArrowLeft size={15}/> Laboratory</Link><ModeBadge mode="live-training"/></div><div className="page-intro"><h1>Watch gradient descent learn.</h1><p>Move the learning rate, adjust noise or step through optimization. This experiment computes an actual linear regression model entirely in your browser, with a fixed seed for reproducibility.</p></div><RegressionLab/><div className="quiet-note" style={{marginTop:16}}><strong>Scientific note.</strong> this educational experiment uses a synthetic one-feature dataset and full-batch gradient descent. It does not establish generalization to real data; subsequent labs will add held-out evaluation and matched model comparisons.</div></div>;
}
