import Link from "next/link";
import { ArrowRight, ArrowUpRight, Atom, BookOpenText, FlaskConical, GitBranch, MoveUpRight } from "lucide-react";
import { ModelCard } from "@/components/ModelCard";
import { NetworkArtwork } from "@/components/NetworkArtwork";
import { models } from "@/content/models";

export default function HomePage() {
  return <div className="page-wrap home-page">
    <section className="hero-panel">
      <div className="hero-copy"><div className="eyebrow"><span className="eyebrow-line"/> THE INTERACTIVE MACHINE LEARNING ATLAS</div>
        <h1>Understand every<br/><em>model.</em> Explore every<br/>possibility.</h1>
        <p>An open, research-focused space to discover architectures, unpack the mathematics, and experience algorithms through reproducible experiments.</p>
        <div className="hero-actions"><Link className="button-primary" href="/models">Explore the models <ArrowRight size={16}/></Link><Link className="button-secondary" href="/lab/linear-regression">Open live laboratory <ArrowUpRight size={16}/></Link></div>
        <div className="hero-meta"><span><i/> RESEARCH-ORIENTED</span><span>OPEN KNOWLEDGE / V0.1</span></div>
      </div>
      <NetworkArtwork />
    </section>
    <div className="stat-strip"><div><strong>08</strong><span>CURATED MODELS</span></div><div><strong>04</strong><span>MODEL FAMILIES</span></div><div><strong>01</strong><span>LIVE EXPERIMENT</span></div><div className="stat-message">A structured path from first principles to scientific machine learning. <MoveUpRight size={19}/></div></div>
    <section className="section-block"><div className="section-heading"><div><span className="section-index">01 / DISCOVER</span><h2>Explore model families</h2><p>Move between foundational methods and modern neural architectures.</p></div><Link href="/atlas" className="text-link">Open algorithm atlas <ArrowUpRight size={16}/></Link></div>
      <div className="family-grid">
        {[
          { label: "Classical ML", count: "03 models", icon: GitBranch, text: "The foundations of inference, decisions and optimization.", hue: "sage" },
          { label: "Deep Learning", count: "03 models", icon: Atom, text: "From layers and convolution to contextual attention.", hue: "blue" },
          { label: "Generative AI", count: "01 model", icon: BookOpenText, text: "Understand the processes behind synthetic data.", hue: "coral" },
          { label: "Scientific ML", count: "01 model", icon: FlaskConical, text: "Bring equations and physical constraints into learning.", hue: "gold" }
        ].map(({label,count,icon:Icon,text,hue})=><Link href={`/models?family=${encodeURIComponent(label)}`} className={`family-card family-${hue}`} key={label}><div className="family-card-top"><Icon size={23} strokeWidth={1.5}/><ArrowUpRight size={17}/></div><h3>{label}</h3><p>{text}</p><span>{count}</span></Link>)}
      </div>
    </section>
    <section className="section-block"><div className="section-heading"><div><span className="section-index">02 / THE COLLECTION</span><h2>Selected models</h2><p>Start with the ideas that shape how intelligent systems learn.</p></div><Link href="/models" className="text-link">Browse all eight <ArrowUpRight size={16}/></Link></div>
      <div className="model-grid featured-grid">{[models[0],models[4],models[5]].map((model,i)=><ModelCard key={model.slug} model={model} number={i+1}/>)}</div>
    </section>
    <section className="lab-feature"><div className="lab-feature-icon"><FlaskConical size={26}/></div><div><span className="section-index">03 / EXPERIMENT</span><h2>Don't just read about it. Run it.</h2><p>Watch a real gradient-descent algorithm fit a line, inspect every update, and export a repeatable experiment record.</p></div><Link className="button-primary" href="/lab/linear-regression">Launch first experiment <ArrowRight size={16}/></Link></section>
  </div>;
}
