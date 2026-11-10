"use client";

import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { families, models, type ModelFamily } from "@/content/models";
import { ModelCard } from "@/components/ModelCard";

export function ModelLibrary({ initialFamily }: { initialFamily?: string }) {
  const [query, setQuery] = useState("");
  const [family, setFamily] = useState<ModelFamily | "All">(families.includes(initialFamily as ModelFamily) ? initialFamily as ModelFamily : "All");
  const filtered = useMemo(() => models.filter((model) => {
    const combined = [model.name,model.chineseName,model.acronym,model.summary,model.paradigm,model.family,...model.tags].join(" ").toLowerCase();
    return (family === "All" || model.family === family) && combined.includes(query.trim().toLowerCase());
  }), [family,query]);
  return <>
    <div className="toolbar">
      <label className="search-field"><Search size={16}/><input value={query} onChange={(event)=>setQuery(event.target.value)} placeholder="Search models, methods, tasks…" aria-label="Search models" /></label>
      <div className="filter-list" aria-label="Filter by family">{(["All",...families] as const).map((name)=><button key={name} type="button" className={`filter-pill ${family===name?"selected":""}`} onClick={()=>setFamily(name)} aria-pressed={family===name}>{name}</button>)}</div>
    </div>
    <div className="result-count" aria-live="polite">Showing {filtered.length} of {models.length} models</div>
    {filtered.length ? <div className="model-grid">{filtered.map((model)=><ModelCard key={model.slug} model={model}/>)}</div> : <div className="empty-state">No models match this search. Try a different family or keyword.</div>}
  </>;
}
