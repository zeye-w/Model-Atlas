import Link from "next/link";
import { ArrowRight } from "lucide-react";

export default function NotFound() {
  return <div className="page-wrap"><div className="page-intro"><h1>That page isn’t in the atlas yet.</h1><p>Explore the curated library or return to the overview.</p></div><Link className="button-primary" href="/models">Browse models <ArrowRight size={16}/></Link></div>;
}
