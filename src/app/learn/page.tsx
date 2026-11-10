import type { Metadata } from "next";

export const metadata: Metadata = { title: "Learning Paths" };

const path = [
  ["Mathematical foundations","Vectors, derivatives, loss functions, and optimization."],
  ["Classical machine learning","Regression, distance-based prediction, and decision trees."],
  ["Neural network mechanics","Layers, activations, gradient descent, and backpropagation."],
  ["Modern architectures","Convolution, attention, generative modeling, and multimodality."],
  ["Scientific machine learning","Physics-informed training, inverse problems, and operator learning."],
];

export default function LearnPage() {
  return <div className="page-wrap"><div className="page-intro"><h1>Learn in connected layers.</h1><p>A research-oriented sequence from the mathematics of optimization to the architectures used in modern AI and scientific computing.</p></div><div className="sequence-list">{path.map(([title,description],i)=><div className="sequence-item" key={title}><span className="sequence-number">{i+1}</span><div><h3>{title}</h3><p>{description}</p></div></div>)}</div><div className="quiet-note" style={{marginTop:16}}>This is a curriculum outline for Phase 1; full lessons and progress tracking are planned for later releases.</div></div>;
}
