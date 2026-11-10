import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { ModeBadge } from "@/components/ModelCard";
import { RegressionLab } from "@/components/RegressionLab/RegressionLab";
import "./regression.css";

export const metadata: Metadata = {
  title: "Regression Laboratory",
  description: "Train AdaBoost, Gradient Boosting, Random Forest, CatBoost-style boosting and an RBF network on the same data in your browser, replay how each fit is built and compare their errors.",
};

export default function RegressionLabPage() {
  return <div className="page-wrap">
    <div className="detail-topline"><Link href="/lab" className="back-link"><ArrowLeft size={15}/> Laboratory</Link><ModeBadge mode="live-training"/></div>
    <div className="page-intro"><h1>Five ways to fit a curve.</h1><p>Boosting, bagging and a radial basis function network learn the same noisy one-dimensional function. Replay each model as it grows, see what every round adds, and compare their errors on held-out points.</p></div>
    <RegressionLab/>
    <div className="quiet-note" style={{ marginTop: 16 }}><strong>What is real here.</strong> All five algorithms are implemented in this site’s code and train in your browser on seeded data; the tests check tree fitting, residual updates, AdaBoost.R2 weights, the ridge solution and that every model beats predicting the mean. They are compact teaching implementations, not the scikit-learn or CatBoost libraries — each panel says what is simplified.</div>
  </div>;
}
