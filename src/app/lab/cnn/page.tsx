import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { CnnLab } from "@/components/CnnLab/CnnLab";
import "./cnn.css";

export const metadata: Metadata = {
  title: "CNN Interactive Laboratory",
  description: "Inspect cross-correlation and train a reproducible 1D CNN classifier in your browser.",
};

export default function CnnPage() {
  return <div className="page-wrap">
    <div className="detail-topline"><Link href="/lab" className="back-link"><ArrowLeft size={15}/> All laboratories</Link><span className="mode-badge mode-live">LIVE TRAINING</span></div>
    <div className="page-intro"><span className="section-index">LAB 002 / DEEP LEARNING</span><h1>CNN Interactive Laboratory</h1><p>Step through 1D cross-correlation, train a waveform classifier, and inspect the gradients behind each update.</p></div>
    <CnnLab/>
  </div>;
}
