"use client";

import type { PointerEvent } from "react";

/* A layered network drawn as an illustration: many small nodes, hairline
   connections between adjacent layers, and two highlighted paths along which a
   signal travels. Purely decorative (aria-hidden). */
const W = 565, H = 365;
const layers = [5, 8, 11, 9, 6, 2];
const nodes = layers.map((count, l) => {
  const x = 40 + l * ((W - 80) / (layers.length - 1));
  const span = 40 + count * 26;
  return Array.from({ length: count }, (_, i) => ({ x, y: H / 2 + (i - (count - 1) / 2) * (span / Math.max(1, count - 1)) * (count > 1 ? 1 : 0) }));
});
const paths = [[1, 2, 4, 3, 2, 0], [3, 5, 7, 6, 4, 1]];
const onPath = (l: number, i: number) => paths.findIndex((p) => p[l] === i);
const toPoints = (p: number[]) => p.map((i, l) => `${nodes[l][i].x},${nodes[l][i].y}`).join(" ");
/** A stable sprinkle of lightly filled nodes, so the field is not uniform. */
const shaded = (l: number, i: number) => ((l * 7 + i * 13) % 5) === 0;

function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
  if (event.pointerType === "touch" || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const target = event.currentTarget;
  const rect = target.getBoundingClientRect();
  const x = (event.clientX - rect.left) / rect.width - 0.5;
  const y = (event.clientY - rect.top) / rect.height - 0.5;

  target.style.setProperty("--network-rotate-y", `${(x * 7).toFixed(2)}deg`);
  target.style.setProperty("--network-rotate-x", `${(-y * 5).toFixed(2)}deg`);
  target.style.setProperty("--network-shift-x", `${(x * 10).toFixed(1)}px`);
  target.style.setProperty("--network-shift-y", `${(y * 8).toFixed(1)}px`);
  target.style.setProperty("--network-glow-x", `${(x * 24).toFixed(1)}px`);
  target.style.setProperty("--network-glow-y", `${(y * 18).toFixed(1)}px`);
}

function resetPointer(event: PointerEvent<HTMLDivElement>) {
  const target = event.currentTarget;
  [
    "--network-rotate-y", "--network-rotate-x",
    "--network-shift-x", "--network-shift-y",
    "--network-glow-x", "--network-glow-y"
  ].forEach((property) => target.style.removeProperty(property));
}

export function NetworkArtwork() {
  return <div
    className="network-artwork"
    aria-hidden="true"
    onPointerMove={handlePointerMove}
    onPointerLeave={resetPointer}
  >
    <svg className="network-svg" viewBox={`0 0 ${W} ${H}`} fill="none" xmlns="http://www.w3.org/2000/svg">
      <g stroke="#d9d3c6" strokeWidth=".6" strokeOpacity=".7">
        {nodes.slice(0, -1).map((layer, l) => layer.map((a, i) => nodes[l + 1].map((b, j) =>
          <line key={`${l}-${i}-${j}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y}/>)))}
      </g>
      {paths.map((p, k) => <polyline key={`path${k}`} points={toPoints(p)} stroke={k === 0 ? "#c96442" : "#6f8a63"} strokeWidth="1.4" strokeOpacity=".55" strokeLinejoin="round"/>)}
      {paths.map((p, k) => <polyline key={`signal${k}`} points={toPoints(p)} pathLength={100} className={`network-signal network-signal-${k}`} stroke={k === 0 ? "#c96442" : "#6f8a63"}/>)}
      {nodes.map((layer, l) => layer.map(({ x, y }, i) => {
        const k = onPath(l, i);
        return <g key={`${l}-${i}`} className="network-node">
          {k >= 0 && l === layers.length - 1 && <circle className={`network-pulse network-pulse-${k === 0 ? 3 : 6}`} cx={x} cy={y} r="12"/>}
          <circle cx={x} cy={y} r={k >= 0 ? 6 : 4.5}
            fill={k === 0 ? "#c96442" : k === 1 ? "#6f8a63" : shaded(l, i) ? "#ebe6dc" : "#ffffff"}
            stroke={k >= 0 ? "#faf9f5" : "#cfc8ba"} strokeWidth={k >= 0 ? 1.5 : 1}/>
        </g>;
      }))}
    </svg>
  </div>;
}
