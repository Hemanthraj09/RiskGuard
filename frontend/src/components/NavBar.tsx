"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/performance", label: "Model Performance" },
  { href: "/dashboard", label: "Risk Analyst" },
  { href: "/simulate", label: "Simulation Console" },
];

export function NavBar() {
  const pathname = usePathname();

  return (
    <header className="border-b" style={{ borderColor: "var(--border)", background: "var(--surface-raised)" }}>
      <div className="mx-auto flex max-w-7xl items-center gap-8 px-6 py-3.5">
        <span className="text-sm font-semibold tracking-tight" style={{ color: "var(--text-primary)" }}>
          RiskGuard
        </span>
        <nav className="flex gap-1">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className="rounded-md px-3 py-1.5 text-sm font-medium transition-colors"
                style={{
                  color: active ? "var(--series-1)" : "var(--text-secondary)",
                  background: active ? "var(--series-1-soft)" : "transparent",
                }}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
