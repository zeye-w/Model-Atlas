import type { Metadata } from "next";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";

export const metadata: Metadata = { title: "Model Comparison" };

export default function ComparePage() {
  return <div className="page-wrap"><div className="page-intro"><span className="section-index">04 / COMPARE</span><h1>Compare with context.</h1><p>Good comparisons control the dataset, splits, preprocessing and metrics. The first paired live experiments will arrive after the individual laboratories are validated.</p></div>
    <div className="resource-grid"><div className="resource-card"><h3>KNN ↔ Decision Tree</h3><p>Compare local neighborhoods and axis-aligned partitions on the same two-dimensional classification data.</p><Link href="/models/knn">Understand KNN <ArrowUpRight size={14}/></Link></div><div className="resource-card"><h3>Linear Regression ↔ MLP</h3><p>Explore how nonlinear representation changes the fitted function under the same regression task.</p><Link href="/models/linear-regression">Start with regression <ArrowUpRight size={14}/></Link></div></div><div className="quiet-note" style={{marginTop:20}}>Comparison runs are planned, not executable in Phase 1. No benchmark scores are displayed without matched experimental conditions.</div>
  </div>;
}
