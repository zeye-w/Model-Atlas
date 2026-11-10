import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { CnnLab } from "@/components/CnnLab/CnnLab";
import { Formula } from "@/components/Formula";
import { ModeBadge } from "@/components/ModelCard";
import { cnnLessons } from "@/content/cnn-theory";
import "./cnn.css";

export const metadata: Metadata = {
  title: "CNN Interactive Laboratory",
  description: "Inspect cross-correlation and train a reproducible 1D CNN classifier in your browser.",
};

export default function CnnPage() {
  return <div className="page-wrap">
    <div className="detail-topline"><Link href="/lab" className="back-link"><ArrowLeft size={15}/> Laboratory</Link><ModeBadge mode="live-training"/></div>
    <div className="page-intro"><h1>CNN laboratory</h1><p>Step through 1D cross-correlation, train a waveform classifier, and inspect the gradients behind each update.</p></div>
    <CnnLab theoryFormulas={cnnLessons.map((lesson) => <Formula key={lesson.title} className="cnn-formula" latex={lesson.latex} label={lesson.title}/>)}/>
  </div>;
}
