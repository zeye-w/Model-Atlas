import type { Metadata } from "next";
import { SiteShell } from "@/components/SiteShell";
import "@fontsource-variable/inter/wght.css";
import "@fontsource-variable/source-serif-4/opsz.css";
import "@fontsource-variable/source-serif-4/opsz-italic.css";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Model Atlas — Interactive AI Research", template: "%s | Model Atlas" },
  description: "Explore machine learning architectures, mathematics and reproducible interactive experiments.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><a className="skip-link" href="#main-content">Skip to content</a><SiteShell>{children}</SiteShell></body></html>;
}
