import type { Metadata } from "next";
import { ModelLibrary } from "@/components/ModelLibrary";

export const metadata: Metadata = { title: "Model Library" };

export default async function ModelsPage({ searchParams }: { searchParams: Promise<{ family?: string }> }) {
  const { family } = await searchParams;
  return <div className="page-wrap"><div className="page-intro"><span className="section-index">01 / MODEL LIBRARY</span><h1>A field guide to machine learning.</h1><p>Discover fundamental algorithms, neural architectures and scientific models. Every entry connects the intuition, mathematics, architecture and research literature.</p></div><ModelLibrary initialFamily={family}/></div>;
}
