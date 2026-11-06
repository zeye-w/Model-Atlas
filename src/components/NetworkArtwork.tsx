"use client";

import type { PointerEvent } from "react";

const positions = [
  [45, 142], [124, 75], [155, 198], [222, 125], [292, 45], [320, 197], [399, 97], [435, 212], [515, 142],
  [124, 296], [232, 286], [335, 316], [511, 301]
];
const edges = [[0,1],[0,2],[1,3],[2,3],[2,9],[3,4],[3,5],[3,6],[3,10],[4,6],[5,6],[5,10],[5,11],[6,7],[6,8],[7,8],[7,11],[8,12],[9,10],[10,11],[11,12]];

function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
  if (event.pointerType === "touch") return;
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
    <div className="network-glow"/><div className="network-grid"/>
    <svg className="network-svg" viewBox="0 0 565 365" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs><linearGradient id="linkGradient" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#d7a98e"/><stop offset="1" stopColor="#b6a98f"/></linearGradient></defs>
      {edges.map(([a,b],i)=><line key={i} x1={positions[a][0]} y1={positions[a][1]} x2={positions[b][0]} y2={positions[b][1]} stroke="url(#linkGradient)" strokeWidth="1.25" opacity=".63" />)}
      {positions.map(([x,y],i)=><g key={i} className="network-node">
        {(i===3||i===6)&&<circle className={`network-pulse network-pulse-${i}`} cx={x} cy={y} r="31"/>}
        <circle cx={x} cy={y} r={i===3||i===6?25:13} fill={i===3?"#c16b4b":i===6?"#879078":"#fffdf8"} stroke={i===3||i===6?"#fffaf5":"#c8b5a5"} strokeWidth="1.5"/>
        <circle cx={x} cy={y} r={i===3||i===6?8:4} fill={i===3||i===6?"#fffaf5":"#c9a58e"}/>
      </g>)}
      <g transform="translate(181 78)"><rect width="94" height="25" rx="5" fill="#fffdf8" stroke="#e8ddd1"/><text x="47" y="17" textAnchor="middle" fontSize="9" fill="#806a5a" fontFamily="sans-serif">INPUT LAYER</text></g>
      <g transform="translate(348 249)"><rect width="112" height="25" rx="5" fill="#fffdf8" stroke="#e8ddd1"/><text x="56" y="17" textAnchor="middle" fontSize="9" fill="#806a5a" fontFamily="sans-serif">HIDDEN STATES</text></g>
    </svg>
    <span className="artwork-coordinate">FIG 01. / NEURAL REPRESENTATIONS</span>
  </div>;
}
