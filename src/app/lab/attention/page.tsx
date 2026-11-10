import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { AttentionLab } from "@/components/AttentionLab/AttentionLab";
import { Formula } from "@/components/Formula";
import { ModeBadge } from "@/components/ModelCard";
import { attentionPhaseText } from "@/content/attention-phases";
import { phases } from "@/lib/attention";
import "../steps.css";
import "./attention.css";

export const metadata: Metadata = {
  title: "Attention Laboratory",
  description: "Step through single-head scaled dot-product attention on your own sentence: embeddings, queries, keys, values, scores, softmax and the weighted sum, computed in your browser.",
};

export default function AttentionPage() {
  return <div className="page-wrap">
    <div className="detail-topline"><Link href="/lab" className="back-link"><ArrowLeft size={15}/> Laboratory</Link><ModeBadge mode="live-computation"/></div>
    <div className="page-intro"><h1>How a word decides where to look.</h1><p>One attention head, one short sentence. Follow every step from embeddings to the weighted sum, and see how a head built to look one word back differs from one that looks for the same word.</p></div>
    <AttentionLab
      phaseFormulas={phases.map((p) => <Formula key={p.id} className="mlp-formula" latex={attentionPhaseText[p.id].latex} label={p.title}/>)}
      phaseSummaries={phases.map((p) => attentionPhaseText[p.id].summary)}
    />
    <div className="quiet-note" style={{ marginTop: 16 }}><strong>What is real here.</strong> Every matrix on this page is computed in your browser from the sentence you typed. The weights are not trained: two heads are constructed by hand (their behaviour is checked by tests) and one is random, as at initialisation. A full Transformer adds many heads, learned weights, residual connections, layer normalisation and feed-forward layers.</div>
  </div>;
}
