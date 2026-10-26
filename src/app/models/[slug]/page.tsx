import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, ArrowRight, ArrowUpRight, CheckCircle2, CircleDashed, ExternalLink } from "lucide-react";
import { getModel, models } from "@/content/models";
import { ModeBadge } from "@/components/ModelCard";

export function generateStaticParams() { return models.map((model)=>({slug:model.slug})); }

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const model = getModel(slug);
  return { title: model?.name ?? "Model", description: model?.summary };
}

export default async function ModelDetailPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const model = getModel(slug);
  if (!model) notFound();
  const live = model.executionMode === "live-training" || model.executionMode === "live-computation";
  return <div className="page-wrap">
    <div className="detail-topline"><Link href="/models" className="back-link"><ArrowLeft size={15}/> Back to model library</Link><ModeBadge mode={model.executionMode}/></div>
    <header id="overview" className="detail-header"><div><span className="section-index">MODEL / {model.family.toUpperCase()}</span><h1>{model.name}</h1><div className="detail-subtitle">{model.chineseName} · {model.paradigm}</div><p>{model.overview}</p><div className="tag-row">{model.tags.map((tag)=><span className="tag" key={tag}>{tag}</span>)}</div></div><div className="detail-symbol" style={{"--model-color":model.color} as React.CSSProperties}>{model.acronym}</div></header>
    <nav className="detail-tabs" aria-label="Model sections"><a href="#overview">Overview</a><a href="#architecture">Architecture</a><a href="#mathematics">Mathematics</a><a href="#playground">Playground</a><a href="#research">Research</a></nav>
    <div className="detail-grid">
      <div>
        <section id="architecture" className="detail-panel"><span className="panel-kicker">01 / ARCHITECTURE</span><h2>How information moves</h2><div className="architecture-flow">{model.architecture.map((step,i)=><div key={step} style={{display:"flex",alignItems:"center",gap:8}}><div className="architecture-step">{String(i+1).padStart(2,"0")} · {step}</div>{i<model.architecture.length-1&&<ArrowRight className="architecture-arrow" size={15}/>}</div>)}</div><p>These stages summarize the model conceptually; the exact implementation varies by model variant and task.</p></section>
        <section id="mathematics" className="detail-panel" style={{marginTop:18}}><span className="panel-kicker">02 / MATHEMATICS</span><h2>Core formulation</h2><div className="formula-box">{model.formula}</div><p>{model.formulaDescription}</p><div id="playground" className="detail-action"><div><strong style={{fontSize:12}}>{live ? "Run the equation, not just the animation." : "Interactive laboratory in the roadmap"}</strong><p>{live ? "Change real parameters and inspect the training loss." : "This model is documented; interactive computation is not shipped yet."}</p></div>{live ? <Link className="button-primary" href={`/lab/${model.slug}`}>Open lab <ArrowUpRight size={14}/></Link> : <Link className="button-secondary" href="/lab">Lab roadmap <ArrowRight size={14}/></Link>}</div></section>
      </div>
      <div>
        <section className="detail-panel"><span className="panel-kicker">03 / RESEARCH NOTES</span><h2>Where it works</h2><ul>{model.useCases.map((s)=><li key={s}>{s}</li>)}</ul><h2 style={{marginTop:29}}>Strengths</h2><ul>{model.strengths.map((s)=><li key={s}><CheckCircle2 size={13} style={{display:"inline",verticalAlign:"-2px",marginRight:6,color:"#6c9b8e"}}/>{s}</li>)}</ul><h2 style={{marginTop:29}}>Limitations</h2><ul>{model.limitations.map((s)=><li key={s}><CircleDashed size={13} style={{display:"inline",verticalAlign:"-2px",marginRight:6,color:"#bd9982"}}/>{s}</li>)}</ul></section>
        <section id="research" className="detail-panel" style={{marginTop:18}}><span className="panel-kicker">04 / REFERENCES</span><h2>Further reading</h2><a className="paper-link" href={model.paper.url} target="_blank" rel="noreferrer"><ExternalLink size={15}/><span>{model.paper.name}<br/><small style={{color:"#a1b0b5"}}>{model.paper.year} · External source</small></span></a><h2 style={{marginTop:30}}>Connected models</h2><div className="related-list">{model.related.map((id)=>{const related=getModel(id);return related?<Link key={id} href={`/models/${id}`}>{related.name} ↗</Link>:null;})}</div></section>
      </div>
    </div>
  </div>;
}
