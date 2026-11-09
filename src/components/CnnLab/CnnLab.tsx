"use client";

import { useState, type ReactNode } from "react";
import { Explorer } from "./Explorer";
import { Training } from "./Training";
import { Theory } from "./Theory";

const tabs = ["CNN Explorer", "CNN Training Bench", "Theory"];

function exportJson(value: object) {
  const data = { schemaVersion: "model-atlas.cnn.v1", modelId: "cnn", ...value };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "model-atlas-cnn.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function CnnLab({ theoryFormulas }: { theoryFormulas: ReactNode[] }) {
  const [active, setActive] = useState(0);
  const [visitedTraining, visitTraining] = useState(false);
  function open(workspace: number) {
    setActive(workspace);
    if (workspace === 1) visitTraining(true);
    document.getElementById("cnn-tab-" + workspace)?.focus();
  }
  return <div className="cnn-lab">
    <div className="cnn-tabs" role="tablist" aria-label="CNN workspaces">
      {tabs.map((name, i) => <button key={name} id={"cnn-tab-" + i} role="tab" aria-selected={active === i}
        aria-controls={"cnn-panel-" + i} tabIndex={active === i ? 0 : -1} onClick={() => open(i)}
        onKeyDown={event => {
          const keys: Record<string, number> = { ArrowRight: (i + 1) % 3, ArrowLeft: (i + 2) % 3, Home: 0, End: 2 };
          if (event.key in keys) { event.preventDefault(); open(keys[event.key]); }
        }}>{name}</button>)}
    </div>
    <div id="cnn-panel-0" role="tabpanel" aria-labelledby="cnn-tab-0" hidden={active !== 0}><Explorer exportJson={exportJson}/></div>
    <div id="cnn-panel-1" role="tabpanel" aria-labelledby="cnn-tab-1" hidden={active !== 1}>{visitedTraining && <Training exportJson={exportJson} active={active === 1}/>}</div>
    <div id="cnn-panel-2" role="tabpanel" aria-labelledby="cnn-tab-2" hidden={active !== 2}><Theory open={open} formulas={theoryFormulas}/></div>
  </div>;
}
