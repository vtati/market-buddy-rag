"use client";

import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { ChevronDown, FileText } from "lucide-react";
import type { Citation } from "@/lib/api";
import { cn } from "@/lib/utils";

/** Expandable evidence card: the citation the answer rests on.
 *  The doc · section label carries the gold citation highlight —
 *  the one place gold marks evidence. */
export function EvidenceCard({ citation, index }: { citation: Citation; index: number }) {
  const [open, setOpen] = useState(index === 0);
  const reduce = useReducedMotion();
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-popover">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex min-h-[44px] w-full cursor-pointer items-center gap-3 px-4 py-2.5 text-left transition-colors hover:bg-accent"
      >
        <FileText className="h-4 w-4 shrink-0 text-accent-text" />
        <span className="min-w-0 flex-1 truncate text-[13px]">
          <span className="font-medium text-accent-text">{citation.doc}</span>
          <span className="text-muted-foreground"> · {citation.section}</span>
        </span>
        <span className="tnum hidden font-mono text-[11px] text-muted-foreground sm:block">
          chars {citation.span[0].toLocaleString()}–{citation.span[1].toLocaleString()}
        </span>
        <ChevronDown
          className={cn(
            "h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-200",
            open && "rotate-180",
          )}
        />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={reduce ? { height: "auto" } : { height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={reduce ? { height: "auto", opacity: 0 } : { height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
          >
            <p className="border-t border-border px-4 py-3 text-[13px] leading-relaxed text-foreground/85">
              {citation.excerpt}
              <span className="text-muted-foreground"> …</span>
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
