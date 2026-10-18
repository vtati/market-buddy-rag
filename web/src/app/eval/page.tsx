"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { ChevronDown, CircleAlert } from "lucide-react";
import { fetchEval, type EvalResponse, type EvalRow } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { NumberTicker } from "@/components/number-ticker";
import { Reveal } from "@/components/reveal";
import { cn } from "@/lib/utils";

function MetricCard({
  label,
  children,
  sub,
  delay,
}: {
  label: string;
  children: React.ReactNode;
  sub: string;
  delay: number;
}) {
  return (
    <Reveal delay={delay}>
      <Card>
        <CardContent className="pt-6">
          <p className="text-[12px] font-medium tracking-wide text-muted-foreground">
            {label}
          </p>
          <p className="tnum mt-2 font-display text-[40px] leading-none font-medium text-accent-text">
            {children}
          </p>
          <p className="mt-2.5 text-[12.5px] text-muted-foreground">{sub}</p>
        </CardContent>
      </Card>
    </Reveal>
  );
}

function RowDetail({ row }: { row: EvalRow }) {
  return (
    <div className="space-y-3 px-4 py-4">
      <p className="text-[13px] text-muted-foreground">
        <span className="font-medium text-foreground">Expected:</span> {row.expected}
      </p>
      <p className="text-[13px] text-muted-foreground">
        <span className="font-medium text-foreground">Judge:</span> {row.note}
      </p>
      <div className="rounded-lg border border-border bg-popover px-4 py-3">
        <p className="text-[13.5px] leading-relaxed whitespace-pre-line text-foreground/90">
          {row.answer}
        </p>
      </div>
      {row.citations.length > 0 && (
        <p className="text-[12.5px] text-muted-foreground">
          Cited:{" "}
          {row.citations.map((c) => `${c.doc} · ${c.section}`).join("  ·  ")}
        </p>
      )}
      <p className="tnum font-mono text-[11.5px] text-muted-foreground">
        faithfulness: {row.faithfulness_detail}
      </p>
    </div>
  );
}

export default function EvalPage() {
  const [data, setData] = useState<EvalResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const reduce = useReducedMotion();

  useEffect(() => {
    fetchEval().then(setData).catch((e: Error) => setError(e.message));
  }, []);

  const failures = data?.rows.filter((r) => !r.pass) ?? [];
  const e11 = data?.rows.find((r) => r.id === "E11");

  return (
    <div className="py-8 lg:py-10">
      <Reveal>
        <h1 className="font-display text-4xl font-medium tracking-tight sm:text-[44px]">
          Eval Dashboard<span className="text-accent-text">.</span>
        </h1>
        <p className="mt-3 max-w-3xl text-[15px] leading-relaxed text-muted-foreground">
          Fifteen questions run end-to-end through the same graph the chat page
          uses — nine answerable, two cross-company comparisons, two ambiguous,
          two unanswerable. Each is judged mechanically: gold-section citations,
          every figure re-verified, refusals where refusal is the only honest
          answer.
        </p>
        <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-muted-foreground">
          Measured on the bundled corpus (AAPL · MSFT · NVDA snapshot, Oct
          2026). Session-ingested companies are not part of these
          measurements.
        </p>
        {data && (
          <p className="tnum mt-2 font-mono text-[12px] text-muted-foreground">
            Measured {new Date(data.generated_at).toLocaleString()} · regenerate
            with <span className="font-mono">python -m src.evaluate</span>
          </p>
        )}
      </Reveal>

      {error && (
        <p className="mt-6 flex items-start gap-2 text-sm text-loss-text">
          <CircleAlert className="mt-0.5 h-4 w-4 shrink-0" />
          Could not load eval results ({error}). Run
          {" "}<code className="font-mono">python -m src.evaluate</code> and start
          the API on port 8000.
        </p>
      )}

      {data && (
        <>
          <div className="mt-8 grid gap-6 sm:grid-cols-3">
            <MetricCard
              label="Pass rate"
              sub={`${data.passed} of ${data.total} questions behaved as specified`}
              delay={0.05}
            >
              <NumberTicker value={data.pass_rate * 100} format={(v) => `${v.toFixed(0)}%`} />
            </MetricCard>
            <MetricCard
              label="Faithfulness"
              sub="Answered questions whose every figure re-verified against the cited chunks"
              delay={0.1}
            >
              <NumberTicker
                value={data.faithfulness_rate_answered * 100}
                format={(v) => `${v.toFixed(0)}%`}
              />
            </MetricCard>
            <MetricCard
              label="Mean latency"
              sub="End-to-end per question, local CPU (retrieval + rerank + verify)"
              delay={0.15}
            >
              <NumberTicker
                value={data.mean_latency_ms / 1000}
                format={(v) => `${v.toFixed(1)}s`}
              />
            </MetricCard>
          </div>

          <Reveal delay={0.2}>
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Per-question results</CardTitle>
                <p className="text-[12.5px] text-muted-foreground">
                  Select a row for the recorded answer, citations, and the
                  judge&apos;s note.
                </p>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[720px] text-left text-[13.5px]">
                    <thead>
                      <tr className="border-b border-border text-[12px] tracking-wide text-muted-foreground">
                        <th className="py-2 pr-3 font-medium">#</th>
                        <th className="py-2 pr-3 font-medium">Type</th>
                        <th className="py-2 pr-3 font-medium">Question</th>
                        <th className="py-2 pr-3 font-medium">Outcome</th>
                        <th className="tnum py-2 pr-3 text-right font-medium">Latency</th>
                        <th className="py-2 text-right font-medium">Verdict</th>
                        <th className="py-2 pl-2" aria-label="Expand" />
                      </tr>
                    </thead>
                    <tbody>
                      {data.rows.map((row) => {
                        const open = openId === row.id;
                        return [
                          <tr
                            key={row.id}
                            onClick={() => setOpenId(open ? null : row.id)}
                            className="cursor-pointer border-b border-border/60 transition-colors hover:bg-accent/60"
                          >
                            <td className="tnum py-3 pr-3 font-mono text-[12px] text-muted-foreground">
                              {row.id}
                            </td>
                            <td className="py-3 pr-3">
                              <Badge variant="muted" className="capitalize">
                                {row.type}
                              </Badge>
                            </td>
                            <td className="max-w-[380px] py-3 pr-3">{row.question}</td>
                            <td className="py-3 pr-3 text-muted-foreground">
                              {row.refused ? "Refused" : "Answered"}
                            </td>
                            <td className="tnum py-3 pr-3 text-right font-mono text-[12.5px]">
                              {(row.latency_ms / 1000).toFixed(1)}s
                            </td>
                            <td className="py-3 text-right">
                              <Badge variant={row.pass ? "gain" : "loss"}>
                                {row.pass ? "PASS" : "FAIL"}
                              </Badge>
                            </td>
                            <td className="py-3 pl-2">
                              <ChevronDown
                                className={cn(
                                  "h-4 w-4 text-muted-foreground transition-transform duration-200",
                                  open && "rotate-180",
                                )}
                              />
                            </td>
                          </tr>,
                          open && (
                            <tr key={`${row.id}-detail`} className="border-b border-border/60">
                              <td colSpan={7} className="bg-card/50 p-0">
                                <AnimatePresence initial={false}>
                                  <motion.div
                                    initial={reduce ? { opacity: 0 } : { height: 0, opacity: 0 }}
                                    animate={{ height: "auto", opacity: 1 }}
                                    transition={{ duration: 0.25 }}
                                  >
                                    <RowDetail row={row} />
                                  </motion.div>
                                </AnimatePresence>
                              </td>
                            </tr>
                          ),
                        ];
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </Reveal>

          <Reveal delay={0.25}>
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Failure analysis</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-[13.5px] leading-relaxed text-muted-foreground">
                {failures.length === 0 ? (
                  <>
                    <p>
                      <span className="font-medium text-foreground">
                        No failures in this run.
                      </span>{" "}
                      The road here was not straight — the first measured run
                      passed 8 of 15. Absolute cross-encoder scores refused
                      correct Item 1A evidence (a gold chunk scored −4.8 while
                      cross-company noise scored +2.2), a single fused ranking
                      crowded Microsoft out of a comparison pool entirely, and
                      exact-token sentence matching starved the answer composer.
                      Each failure got a mechanism-level fix, documented in{" "}
                      <span className="font-mono text-[12.5px]">
                        docs/project-documentation.md
                      </span>
                      .
                    </p>
                    {e11 && !e11.answer.includes("215,938") && (
                      <p>
                        <span className="font-medium text-foreground">
                          Known soft spot (E11):
                        </span>{" "}
                        the extractive composer states Apple&apos;s $416.2B total
                        and cites NVIDIA&apos;s Item 7, but does not surface
                        NVIDIA&apos;s $215,938M total-revenue line — it sits in a
                        table chunk ranked just below the merged evidence cut.
                        Every figure it does state verifies; the LLM generation
                        mode composes the explicit head-to-head.
                      </p>
                    )}
                    <p>
                      Ambiguous questions are handled deliberately: no company
                      named in a three-company corpus routes to a clarification
                      (or an honest refusal) — never a silent guess at one
                      company&apos;s filing.
                    </p>
                  </>
                ) : (
                  failures.map((f) => (
                    <p key={f.id}>
                      <span className="font-medium text-loss-text">
                        {f.id} — {f.type}:
                      </span>{" "}
                      {f.question} Expected: {f.expected} Observed: {f.note}
                    </p>
                  ))
                )}
              </CardContent>
            </Card>
          </Reveal>
        </>
      )}
    </div>
  );
}
