"use client";

import { motion, useReducedMotion } from "motion/react";
import { Check, CircleDashed, LoaderCircle } from "lucide-react";
import type { TraceStep } from "@/lib/api";
import { cn } from "@/lib/utils";

const NODE_LABELS: Record<string, string> = {
  contextualize: "Understanding follow-up",
  scope: "Scoping",
  retrieve: "Retrieving",
  grade: "Grading evidence",
  rewrite: "Rewriting query",
  generate: "Generating",
  verify: "Verifying figures",
  refuse: "Refusing",
  clarify: "Clarifying",
};

/** The graph's real step trace, revealed step by step: node name,
 *  measured latency, and the node's own detail string. */
export function StepTrail({ trace }: { trace: TraceStep[] }) {
  const reduce = useReducedMotion();
  return (
    <ol className="relative ml-1.5 space-y-3 border-l border-border pl-5">
      {trace.map((step, i) => {
        const isRefusal = step.node === "refuse";
        const isClarify = step.node === "clarify";
        return (
          <motion.li
            key={`${step.node}-${i}`}
            className="relative"
            initial={reduce ? false : { opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: reduce ? 0 : i * 0.22, duration: 0.3 }}
          >
            <span
              className={cn(
                "absolute top-0.5 -left-[27px] flex h-3.5 w-3.5 items-center justify-center rounded-full border bg-background",
                isRefusal
                  ? "border-loss/60 text-loss-text"
                  : isClarify
                    ? "border-primary/60 text-accent-text"
                    : "border-primary/50 text-accent-text",
              )}
            >
              <Check className="h-2.5 w-2.5" strokeWidth={3} />
            </span>
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
              <span className="text-[13px] font-medium text-foreground">
                {NODE_LABELS[step.node] ?? step.node}
              </span>
              <span className="tnum font-mono text-[11px] text-muted-foreground">
                {step.latency_ms.toFixed(0)} ms
              </span>
            </div>
            <p className="mt-0.5 max-w-prose text-[12.5px] leading-relaxed text-muted-foreground">
              {step.detail}
            </p>
          </motion.li>
        );
      })}
    </ol>
  );
}

const PENDING_STAGES = [
  "Scoping the question",
  "Retrieving from the filings",
  "Grading the evidence",
  "Generating a grounded answer",
  "Verifying every figure",
];

/** Animated placeholder while the graph runs: cycles the stage names
 *  so a several-second local rerank never looks frozen. */
export function PendingTrail({ stage }: { stage: number }) {
  const reduce = useReducedMotion();
  return (
    <ol className="relative ml-1.5 space-y-3 border-l border-border pl-5">
      {PENDING_STAGES.map((label, i) => {
        const done = i < stage;
        const active = i === stage;
        return (
          <li key={label} className="relative">
            <span
              className={cn(
                "absolute top-0.5 -left-[27px] flex h-3.5 w-3.5 items-center justify-center rounded-full border bg-background",
                done
                  ? "border-primary/50 text-accent-text"
                  : active
                    ? "border-primary text-accent-text"
                    : "border-border text-muted-foreground",
              )}
            >
              {done ? (
                <Check className="h-2.5 w-2.5" strokeWidth={3} />
              ) : active ? (
                <LoaderCircle
                  className={cn("h-2.5 w-2.5", !reduce && "animate-spin")}
                />
              ) : (
                <CircleDashed className="h-2.5 w-2.5" />
              )}
            </span>
            <span
              className={cn(
                "text-[13px]",
                active
                  ? "font-medium text-foreground"
                  : done
                    ? "text-muted-foreground"
                    : "text-muted-foreground/60",
              )}
            >
              {label}
              {active ? "…" : ""}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
