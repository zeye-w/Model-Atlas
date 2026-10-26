import Link from "next/link";

export default function NotFound() { return <div className="page-wrap"><div className="page-intro"><span className="section-index">404 / NOT FOUND</span><h1>That model isn't in the atlas yet.</h1><p>Explore the curated library or return to the project overview.</p></div><Link className="button-primary" href="/models">Browse models →</Link></div>; }
