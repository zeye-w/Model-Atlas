import Link from "next/link";
import { ArrowRight, ArrowUpRight, Atom, BookOpenText, FlaskConical, GitBranch } from "lucide-react";
import { ModelCard } from "@/components/ModelCard";
import { NetworkArtwork } from "@/components/NetworkArtwork";
import { familyDescriptions, families, featuredModels, liveLabs, modelsInFamily, models, type ModelFamily } from "@/content/models";

const familyStyle: Record<ModelFamily, { icon: typeof GitBranch; hue: string }> = {
  "Classical ML": { icon: GitBranch, hue: "sage" },
  "Deep Learning": { icon: Atom, hue: "blue" },
  "Generative AI": { icon: BookOpenText, hue: "coral" },
  "Scientific ML": { icon: FlaskConical, hue: "gold" },
};

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export default function HomePage() {
  return <div className="page-wrap home-page">
    <section className="hero-panel">
      <div className="hero-copy">
        <h1>A clearer view of how <em>models</em> learn.</h1>
        <p>An open, research-focused space to discover architectures, unpack the mathematics, and run algorithms as reproducible experiments in your browser.</p>
        <div className="hero-actions">
          <Link className="button-primary" href="/models">Explore the models <ArrowRight size={16}/></Link>
          <Link className="button-secondary" href="/lab">Try a live experiment</Link>
        </div>
        <div className="hero-facts">
          <span><strong>{models.length}</strong> curated models</span>
          <span><strong>{families.length}</strong> model families</span>
          <span><strong>{liveLabs.length}</strong> live {liveLabs.length === 1 ? "experiment" : "experiments"}</span>
        </div>
      </div>
      <NetworkArtwork />
    </section>

    <section className="section-block">
      <div className="section-heading">
        <div><h2>Explore model families</h2><p>Move between foundational methods and modern neural architectures.</p></div>
        <Link href="/atlas" className="text-link">Open the atlas <ArrowUpRight size={15}/></Link>
      </div>
      <div className="family-grid">
        {families.map((label) => {
          const { icon: Icon, hue } = familyStyle[label];
          return <Link href={`/models?family=${encodeURIComponent(label)}`} className={`family-card family-${hue}`} key={label}>
            <span className="family-icon"><Icon size={19} strokeWidth={1.7}/></span>
            <h3>{label}</h3>
            <p>{familyDescriptions[label]}</p>
            <span className="family-count">{plural(modelsInFamily(label).length, "model")}</span>
          </Link>;
        })}
      </div>
    </section>

    <section className="section-block">
      <div className="section-heading">
        <div><h2>Selected models</h2><p>Start with the ideas that shape how intelligent systems learn.</p></div>
        <Link href="/models" className="text-link">Browse all {models.length} <ArrowUpRight size={15}/></Link>
      </div>
      <div className="model-grid">{featuredModels.map((model) => <ModelCard key={model.slug} model={model}/>)}</div>
    </section>

    <section className="lab-feature">
      <div className="lab-feature-icon"><FlaskConical size={22} strokeWidth={1.7}/></div>
      <div><h2>Don’t just read about it. Run it.</h2><p>Watch real gradient descent fit a line, inspect every update, and export a repeatable experiment record.</p></div>
      <Link className="button-primary" href="/lab/linear-regression">Launch first experiment <ArrowRight size={16}/></Link>
    </section>
  </div>;
}
