"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ArrowUpRight, Atom, BookOpenText, Boxes, Compass, FlaskConical, GitCompareArrows, Github, GraduationCap, Menu, Search, Waypoints, X } from "lucide-react";
import { useEffect, useState } from "react";
import { buildCommitUrl, buildSha } from "@/lib/build-info";

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
      <span className="brand-emblem"><Waypoints size={17} strokeWidth={1.8} /></span>
      <span><strong>Model Atlas</strong><small>How models learn</small></span>
    </Link>
    <Link className="sidebar-search" href="/models" onClick={onNavigate}><Search size={15}/> Search models</Link>
    <nav className="sidebar-nav" aria-label="Main navigation">
      {navigation.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(href + "/");
        return <Link onClick={onNavigate} className={`nav-link ${active ? "active" : ""}`} href={href} key={href} aria-current={active ? "page" : undefined}>
          <Icon size={17} strokeWidth={1.75} /><span>{label}</span>
        </Link>;
      })}
    </nav>
    <div className="sidebar-footer">
      <p>An open, research-oriented space to study how intelligent systems learn.</p>
      <a className="sidebar-link" href="https://github.com/zeye-w/Model-Atlas" target="_blank" rel="noreferrer"><Github size={15}/> View on GitHub <ArrowUpRight size={13}/></a>
      <div className="sidebar-note"><span className="status-pulse"/><span>Version 0.1 · <a className="build-sha" href={buildCommitUrl} target="_blank" rel="noreferrer" title="Commit this build was made from"><code>{buildSha}</code></a></span></div>
    </div>
  </>;
}

export function SiteShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => { setMenuOpen(false); }, [pathname]);

  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") setMenuOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menuOpen]);

  useEffect(() => {
    const targets = document.querySelectorAll<HTMLElement>(
      ".page-wrap > *, .family-card, .model-card, .lab-list-card, .resource-card, .atlas-family"
    );
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion || !("IntersectionObserver" in window)) {
      targets.forEach((target) => target.classList.add("is-visible"));
      return;
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.08, rootMargin: "0px 0px -36px 0px" });

    targets.forEach((target) => {
      target.classList.add("motion-pending");
      observer.observe(target);
    });
    return () => observer.disconnect();
  }, [pathname]);

  return <div className="app-shell">
    {menuOpen && <div className="mobile-overlay" onClick={() => setMenuOpen(false)} />}
    {/* One sidebar for every viewport: sticky column on desktop, off-canvas drawer below 900px (see .site-sidebar). */}
    <aside id="site-sidebar" className={`sidebar site-sidebar ${menuOpen ? "open" : ""}`}><SidebarContent onNavigate={() => setMenuOpen(false)}/></aside>
    <div className="main-shell">
      <header className="topbar">
        <div className="topbar-left">
          <button className="icon-button mobile-menu-trigger" onClick={() => setMenuOpen((v) => !v)} aria-label={menuOpen ? "Close navigation" : "Open navigation"} aria-expanded={menuOpen} aria-controls="site-sidebar">{menuOpen ? <X size={20}/> : <Menu size={20}/>}</button>
          <Link className="topbar-brand" href="/">Model Atlas</Link>
        </div>
        <Link className="icon-button" href="/models" aria-label="Search models"><Search size={18}/></Link>
      </header>
      <main id="main-content" className="main-content">{children}</main>
      <footer className="site-footer"><span>© 2026 Model Atlas · An open research project</span><span>Built for curiosity, grounded in mathematics.</span></footer>
    </div>
  </div>;
}
