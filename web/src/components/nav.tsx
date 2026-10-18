"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Building2, FlaskConical, Gauge, MessagesSquare } from "lucide-react";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "Chat", icon: MessagesSquare },
  { href: "/ingest", label: "Add a Company", icon: Building2 },
  { href: "/lab", label: "Chunking Lab", icon: FlaskConical },
  { href: "/eval", label: "Eval Dashboard", icon: Gauge },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/90 backdrop-blur-sm">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-6 px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-baseline gap-2.5">
          <span className="font-display text-[19px] font-semibold tracking-tight">
            Market Buddy
            <span className="text-accent-text"> · RAG</span>
          </span>
          <span className="hidden text-[11px] tracking-wide text-muted-foreground md:block">
            Financial Document Intelligence
          </span>
        </Link>
        <nav className="ml-auto flex items-center gap-1" aria-label="Primary">
          {LINKS.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "flex min-h-[44px] items-center gap-2 rounded-md px-3 text-sm transition-colors duration-200",
                  active
                    ? "font-medium text-accent-text"
                    : "text-muted-foreground hover:bg-accent hover:text-foreground",
                )}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden sm:inline">{label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
