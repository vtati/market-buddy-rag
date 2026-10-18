"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CircleAlert } from "lucide-react";
import { fetchComparison, type ComparisonResponse } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Reveal } from "@/components/reveal";

const GOLD = "#C9A96A";
const NEUTRAL = "#4A5160";
const DEFAULT_CONFIG = "Section · Hybrid+rerank";

function configLabel(strategy: string, mode: string) {
  const s = strategy === "section" ? "Section" : "Fixed";
  const m =
    mode === "vector" ? "Vector" : mode === "hybrid" ? "Hybrid" : "Hybrid+rerank";
  return `${s} · ${m}`;
}

function MetricChart({
  title,
  takeaway,
  data,
  dataKey,
  format,
  domain,
}: {
  title: string;
  takeaway: string;
  data: { name: string; value: number }[];
  dataKey: string;
  format: (v: number) => string;
  domain: [number, number] | undefined;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[240px] w-full" role="img" aria-label={`${title} bar chart`}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 40 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.07)" vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fill: "#9BA1A8", fontSize: 10.5 }}
                interval={0}
                angle={-28}
                textAnchor="end"
                axisLine={{ stroke: "rgba(255,255,255,0.07)" }}
                tickLine={false}
              />
              <YAxis
                domain={domain}
                tick={{ fill: "#9BA1A8", fontSize: 11 }}
                tickFormatter={format}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                formatter={(v) => [format(Number(v)), title]}
                contentStyle={{
                  background: "#171C25",
                  border: "1px solid rgba(255,255,255,0.07)",
                  borderRadius: 10,
                  color: "#E8E6DF",
                  fontSize: 12.5,
                }}
                cursor={{ fill: "rgba(255,255,255,0.04)" }}
              />
              <Bar dataKey={dataKey} radius={[4, 4, 0, 0]} maxBarSize={44}>
                {data.map((d) => (
                  <Cell
                    key={d.name}
                    fill={d.name === DEFAULT_CONFIG ? GOLD : NEUTRAL}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 border-t border-border pt-3 text-[13px] leading-relaxed text-muted-foreground">
          {takeaway}
        </p>
      </CardContent>
    </Card>
  );
}

export default function LabPage() {
  const [data, setData] = useState<ComparisonResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchComparison().then(setData).catch((e: Error) => setError(e.message));
  }, []);

  const rows = data?.rows ?? [];
  const hit = rows.map((r) => ({ name: configLabel(r.strategy, r.mode), value: r.hit_at_4 }));
  const mrr = rows.map((r) => ({ name: configLabel(r.strategy, r.mode), value: r.mrr }));
  const lat = rows.map((r) => ({ name: configLabel(r.strategy, r.mode), value: r.mean_latency_ms }));

  return (
    <div className="py-8 lg:py-10">
      <Reveal>
        <h1 className="font-display text-4xl font-medium tracking-tight sm:text-[44px]">
          The Chunking Lab<span className="text-accent-text">.</span>
        </h1>
        <p className="mt-3 max-w-3xl text-[15px] leading-relaxed text-muted-foreground">
          Two chunking strategies × three retrieval modes, run against 12
          questions whose gold filing sections are known. Hit-rate@4 asks
          whether a chunk from the right section reached the top four; MRR
          rewards it landing first; latency is the mean per question on this
          machine. Gold marks the app&apos;s default configuration.
        </p>
        <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-muted-foreground">
          Measured on the bundled corpus (AAPL · MSFT · NVDA snapshot, Oct
          2026). Session-ingested companies are not part of these
          measurements.
        </p>
        {data && (
          <p className="tnum mt-2 font-mono text-[12px] text-muted-foreground">
            Measured {new Date(data.generated_at).toLocaleString()} · embeddings{" "}
            {data.embedding_model} · reranker {data.reranker_model}
          </p>
        )}
      </Reveal>

      {error && (
        <p className="mt-6 flex items-start gap-2 text-sm text-loss-text">
          <CircleAlert className="mt-0.5 h-4 w-4 shrink-0" />
          Could not load comparison results ({error}). Run
          {" "}<code className="font-mono">python -m src.experiments</code> and start
          the API on port 8000.
        </p>
      )}

      {data && (
        <>
          <div className="mt-8 grid gap-6 xl:grid-cols-3">
            <Reveal delay={0.05}>
              <MetricChart
                title="Hit rate @4"
                data={hit}
                dataKey="value"
                format={(v) => v.toFixed(2)}
                domain={[0, 1]}
                takeaway="Section-aware chunking with plain vector retrieval already reaches 1.00 — on a four-document corpus, dense search saturates. Fixed-size trails at 0.92 in every mode."
              />
            </Reveal>
            <Reveal delay={0.1}>
              <MetricChart
                title="MRR"
                data={mrr}
                dataKey="value"
                format={(v) => v.toFixed(2)}
                domain={[0, 1]}
                takeaway="Reranking moves the top of the ranking — where generation actually reads: section-aware MRR climbs 0.74 → 0.93 once the cross-encoder reorders the hybrid pool. Hybrid fusion alone hurt section chunks (BM25 promotes cross-company keyword matches)."
              />
            </Reveal>
            <Reveal delay={0.15}>
              <MetricChart
                title="Mean latency"
                data={lat}
                dataKey="value"
                format={(v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${v.toFixed(0)}ms`)}
                domain={undefined}
                takeaway="The reranker is the bill: ~20 ms without it, 0.8–2.4 s with it on local CPU. Worth paying where answer precision matters — the app defaults to paying it."
              />
            </Reveal>
          </div>

          <Reveal delay={0.2}>
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Full results</CardTitle>
                <p className="text-[12.5px] text-muted-foreground">
                  The same table as <span className="font-mono">docs/comparison.md</span>,
                  written by the experiment run itself — never hand-typed.
                </p>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[560px] text-left text-[13.5px]">
                    <thead>
                      <tr className="border-b border-border text-[12px] tracking-wide text-muted-foreground">
                        <th className="py-2 pr-4 font-medium">Chunking</th>
                        <th className="py-2 pr-4 font-medium">Retrieval</th>
                        <th className="tnum py-2 pr-4 text-right font-medium">Chunks</th>
                        <th className="tnum py-2 pr-4 text-right font-medium">Hit@4</th>
                        <th className="tnum py-2 pr-4 text-right font-medium">MRR</th>
                        <th className="tnum py-2 text-right font-medium">Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => {
                        const isDefault =
                          configLabel(r.strategy, r.mode) === DEFAULT_CONFIG;
                        return (
                          <tr
                            key={`${r.strategy}-${r.mode}`}
                            className="border-b border-border/60 last:border-0"
                          >
                            <td className="py-2.5 pr-4">
                              {r.strategy === "section" ? "Section-aware" : "Fixed-size"}
                              {isDefault && (
                                <span className="ml-2 text-[11px] font-medium text-accent-text">
                                  app default
                                </span>
                              )}
                            </td>
                            <td className="py-2.5 pr-4 text-muted-foreground">{r.mode}</td>
                            <td className="tnum py-2.5 pr-4 text-right font-mono text-[12.5px]">
                              {r.chunks}
                            </td>
                            <td className="tnum py-2.5 pr-4 text-right font-mono text-[12.5px]">
                              {r.hit_at_4.toFixed(2)}
                            </td>
                            <td className="tnum py-2.5 pr-4 text-right font-mono text-[12.5px]">
                              {r.mrr.toFixed(2)}
                            </td>
                            <td className="tnum py-2.5 text-right font-mono text-[12.5px]">
                              {r.mean_latency_ms.toFixed(0)} ms
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="mt-4 grid gap-4 border-t border-border pt-4 text-[13px] leading-relaxed text-muted-foreground md:grid-cols-2">
                  <p>
                    <span className="font-medium text-foreground">How to read this:</span>{" "}
                    hit@4 can tie while MRR separates — reranking rarely changes
                    whether the right chunk is present, it changes whether it is
                    first. The generator and the verifier consume the top of the
                    ranking, so the app optimises for MRR.
                  </p>
                  <p>
                    <span className="font-medium text-foreground">Method:</span>{" "}
                    12 questions with known gold sections (Apple, Microsoft,
                    NVIDIA business / risks / MD&A / statements). Hybrid = RRF
                    (k=60) over BM25 top-10 + vector top-10; rerank = cross-encoder
                    over the fused top-10, top-4 kept.
                  </p>
                </div>
              </CardContent>
            </Card>
          </Reveal>
        </>
      )}
    </div>
  );
}
