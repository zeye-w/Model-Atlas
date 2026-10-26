"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ArrowUpRight, Atom, BookOpenText, Boxes, Compass, FlaskConical, GitCompareArrows, Github, GraduationCap, Menu, Search, X } from "lucide-react";
import { useState } from "react";

const navigation = [
  { href: "/", label: "Overview", icon: Compass },
  { href: "/atlas", label: "Algorithm Atlas", icon: Atom },
  { href: "/models", label: "Model Library", icon: Boxes },
  { href: "/lab", label: "Laboratory", icon: FlaskConical },
  { href: "/compare", label: "Compare", icon: GitCompareArrows },
  { href: "/research", label: "Research", icon: BookOpenText },
  { href: "/learn", label: "Learning Paths", icon: GraduationCap },
];

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return <>
    <Link className="brand" href="/" onClick={onNavigate} aria-label="Model Atlas home">
      <span className="brand-emblem"><Atom size={24} strokeWidth={1.65} /></span>
      <span><strong>MODEL ATLAS</strong><small>RESEARCH WORKSPACE</small></span>
    </Link>
    <div className="sidebar-section-label">WORKSPACE <span>01</span></div>
    <nav className="sidebar-nav" aria-label="Main navigation">
      {navigation.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(href + "/");
        return <Link onClick={onNavigate} className={`nav-link ${active ? "active" : ""}`} href={href} key={href} aria-current={active ? "page" : undefined}>
          <Icon size={18} strokeWidth={1.75} /><span>{label}</span>{active && <span className="nav-dot" />}
        </Link>;
      })}
    </nav>
    <div className="sidebar-footer">
      <div className="sidebar-note"><span className="status-pulse"/><span>Phase 01 · Foundation</span></div>
      <p>Making machine learning understandable, one experiment at a time.</p>
      <a href="https://github.com/zeye-w/Model-Atlas" target="_blank" rel="noreferrer"><Github size={16}/> View on GitHub <ArrowUpRight size={14}/></a>
    </div>
  </>;
}

export function SiteShell({ children }: { children: React.ReactNode }) {
  const [menuOpen, setMenuOpen] = useState(false);
  return <div className="app-shell">
    <aside className="sidebar desktop-sidebar"><SidebarContent /></aside>
    {menuOpen && <div className="mobile-overlay" onClick={() => setMenuOpen(false)} />}
    <aside className={`sidebar mobile-sidebar ${menuOpen ? "open" : ""}`}><SidebarContent onNavigate={() => setMenuOpen(false)}/></aside>
    <div className="main-shell">
      <header className="topbar">
        <div className="topbar-left">
          <button className="icon-button mobile-menu-trigger" onClick={() => setMenuOpen((v) => !v)} aria-label={menuOpen ? "Close navigation" : "Open navigation"}>{menuOpen ? <X size={21}/> : <Menu size={21}/>}</button>
          <span className="topbar-eyebrow">THE MODEL ATLAS PROJECT</span>
          <span className="topbar-separator"/><span className="topbar-location">Explore intelligence, layer by layer.</span>
        </div>
        <div className="topbar-actions">
          <Link className="topbar-search" href="/models"><Search size={15}/> <span>Search models</span><kbd>↗</kbd></Link>
          <span className="topbar-version">ALPHA / 0.1</span>
        </div>
      </header>
      <main id="main-content" className="main-content">{children}</main>
      <footer className="site-footer"><span>© 2026 Model Atlas · An open research project</span><span>Built for curiosity, grounded in mathematics.</span></footer>
    </div>
  </div>;
}
