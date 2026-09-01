"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/performance", label: "Model Performance" },
  { href: "/dashboard", label: "Risk Analyst" },
  { href: "/simulate", label: "Simulation Console" },
];

function ShieldLogo() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M12 2L4 5v6c0 5.25 3.4 9.74 8 11 4.6-1.26 8-5.75 8-11V5l-8-3z"
        fill="var(--accent)"
        fillOpacity="0.18"
        stroke="var(--accent)"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path
        d="M8.5 12.2l2.4 2.4 4.6-4.9"
        stroke="var(--accent)"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function NavBar() {
  const pathname = usePathname();

  return (
    <header className="border-b" style={{ borderColor: "var(--border)", background: "var(--surface-raised)" }}>
      <div className="mx-auto flex max-w-7xl items-center gap-8 px-6 py-3.5">
        <span className="flex items-center gap-2 text-sm font-semibold tracking-tight" style={{ color: "var(--accent)" }}>
          <ShieldLogo />
          RiskGuard
        </span>
        <nav className="flex gap-1">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className="px-3 py-1.5 text-sm font-medium transition-colors"
                style={{
                  color: active ? "var(--accent)" : "var(--text-secondary)",
                  background: active ? "var(--accent-soft)" : "transparent",
                  borderRadius: "6px 6px 0 0",
                  borderBottom: active ? "2px solid var(--accent)" : "2px solid transparent",
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
