import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, ArrowRight, ArrowUpRight, CheckCircle2, CircleDashed, ExternalLink } from "lucide-react";
import { getModel, models } from "@/content/models";
import { ModeBadge } from "@/components/ModelCard";
import { Formula } from "@/components/Formula";

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
    <div className="detail-topline"><Link href="/models" className="back-link"><ArrowLeft size={15}/> Model library</Link><ModeBadge mode={model.executionMode}/></div>
    <header id="overview" className="detail-header">
      <div>
        <h1>{model.name}</h1>
        <div className="detail-subtitle">{model.chineseName} · {model.family} · {model.paradigm}</div>
        <p>{model.overview}</p>
        <div className="tag-row">{model.tags.map((tag)=><span className="tag" key={tag}>{tag}</span>)}</div>
      </div>
      <div className="detail-symbol" style={{"--model-color":model.color} as React.CSSProperties} aria-hidden="true">{model.acronym}</div>
    </header>
    <nav className="detail-tabs" aria-label="Model sections"><a href="#overview">Overview</a><a href="#architecture">Architecture</a><a href="#mathematics">Mathematics</a><a href="#playground">Playground</a><a href="#research">Research</a></nav>
    <div className="detail-grid">
      <div>
        <section id="architecture" className="detail-panel">
          <h2>How information moves</h2>
          <div className="architecture-flow">{model.architecture.map((step,i)=><div key={step}><span className="architecture-step"><span>{i+1}</span>{step}</span>{i<model.architecture.length-1&&<ArrowRight className="architecture-arrow" size={14}/>}</div>)}</div>
          <p>These stages summarize the model conceptually; the exact implementation varies by model variant and task.</p>
        </section>
        <section id="mathematics" className="detail-panel">
          <h2>Core formulation</h2>
          <Formula className="formula-box" latex={model.latex} label={`${model.name} core formulation`}/>
          <p>{model.formulaDescription}</p>
          <div id="playground" className="detail-action">
            <div><strong>{live ? "Run the equation, not just the animation." : "Interactive laboratory on the roadmap"}</strong><p>{live ? "Change real parameters and inspect the training loss." : "This model is documented; interactive computation has not shipped yet."}</p></div>
            {live && model.lab ? <Link className="button-primary" href={model.lab.href}>Open lab <ArrowUpRight size={14}/></Link> : <Link className="button-secondary" href="/lab">Lab roadmap <ArrowRight size={14}/></Link>}
          </div>
        </section>
      </div>
      <div>
        <section className="detail-panel">
          <h2>Research notes</h2>
          <h3>Where it works</h3>
          <ul>{model.useCases.map((s)=><li className="li-plain" key={s}>{s}</li>)}</ul>
          <h3>Strengths</h3>
          <ul>{model.strengths.map((s)=><li className="li-good" key={s}><CheckCircle2 size={15}/>{s}</li>)}</ul>
          <h3>Limitations</h3>
          <ul>{model.limitations.map((s)=><li className="li-limit" key={s}><CircleDashed size={15}/>{s}</li>)}</ul>
        </section>
        <section id="research" className="detail-panel">
          <h2>Further reading</h2>
          <a className="paper-link" href={model.paper.url} target="_blank" rel="noreferrer"><ExternalLink size={15}/><span><span>{model.paper.name}</span><small>{model.paper.year} · External source</small></span></a>
          <h3 style={{marginTop:24}}>Connected models</h3>
          <div className="related-list">{model.related.map((id)=>{const related=getModel(id);return related?<Link key={id} href={`/models/${id}`}>{related.name} <ArrowUpRight size={13}/></Link>:null;})}</div>
        </section>
      </div>
    </div>
  </div>;
}
