const positions = [
  [45, 142], [124, 75], [155, 198], [222, 125], [292, 45], [320, 197], [399, 97], [435, 212], [515, 142],
  [124, 296], [232, 286], [335, 316], [511, 301]
];
const edges = [[0,1],[0,2],[1,3],[2,3],[2,9],[3,4],[3,5],[3,6],[3,10],[4,6],[5,6],[5,10],[5,11],[6,7],[6,8],[7,8],[7,11],[8,12],[9,10],[10,11],[11,12]];

export function NetworkArtwork() {
  return <div className="network-artwork" aria-hidden="true">
    <div className="network-glow"/><div className="network-grid"/>
    <svg className="network-svg" viewBox="0 0 565 365" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs><linearGradient id="linkGradient" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#a8c8d7"/><stop offset="1" stopColor="#a0b5a5"/></linearGradient></defs>
      {edges.map(([a,b],i)=><line key={i} x1={positions[a][0]} y1={positions[a][1]} x2={positions[b][0]} y2={positions[b][1]} stroke="url(#linkGradient)" strokeWidth="1.25" opacity=".63" />)}
      {positions.map(([x,y],i)=><g key={i} className="network-node"><circle cx={x} cy={y} r={i===3||i===6?25:13} fill={i===3?"#648da7":i===6?"#588e86":"#fff"} stroke={i===3||i===6?"#ffffff":"#9bb7be"} strokeWidth="1.5"/><circle cx={x} cy={y} r={i===3||i===6?8:4} fill={i===3||i===6?"#fff":"#9ebcc2"}/></g>)}
      <g transform="translate(181 78)"><rect width="94" height="25" rx="5" fill="white" stroke="#dfe7e7"/><text x="47" y="17" textAnchor="middle" fontSize="9" fill="#526c7b" fontFamily="sans-serif">INPUT LAYER</text></g>
      <g transform="translate(348 249)"><rect width="112" height="25" rx="5" fill="white" stroke="#dfe7e7"/><text x="56" y="17" textAnchor="middle" fontSize="9" fill="#526c7b" fontFamily="sans-serif">HIDDEN STATES</text></g>
    </svg>
    <span className="artwork-coordinate">FIG 01. / NEURAL REPRESENTATIONS</span>
  </div>;
}
