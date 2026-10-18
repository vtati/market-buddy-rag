"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  Building2,
  CheckCircle2,
  Circle,
  CircleAlert,
  Loader2,
  Search,
} from "lucide-react";
import {
  fetchCorpus,
  fetchIngestStatus,
  resolveCompany,
  startIngest,
  type CorpusResponse,
  type IngestCandidate,
  type IngestStatus,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Reveal } from "@/components/reveal";
import { cn } from "@/lib/utils";

const STAGE_LABELS: Record<string, string> = {
  resolve: "Resolve ticker",
  fetch: "Fetch latest 10-K",
  clean: "Clean & section-map",
  chunk: "Chunk — both strategies",
  embed: "Embed & index",
  figures: "Figures table",
  ready: "Ready",
};

const QUICK_PICKS = [
  { label: "Tesla (precached — no network)", q: "Tesla" },
  { label: "Coca-Cola", q: "Coca-Cola" },
  { label: "Walmart", q: "Walmart" },
];

function StageIcon({ status }: { status: string }) {
  if (status === "done")
    return <CheckCircle2 className="h-5 w-5 shrink-0 text-gain-text" />;
  if (status === "active")
    return <Loader2 className="h-5 w-5 shrink-0 animate-spin text-accent-text" />;
  if (status === "error")
    return <CircleAlert className="h-5 w-5 shrink-0 text-loss-text" />;
  return <Circle className="h-5 w-5 shrink-0 text-muted-foreground/50" />;
}

export default function IngestPage() {
  const [query, setQuery] = useState("");
  const [resolving, setResolving] = useState(false);
  const [candidates, setCandidates] = useState<IngestCandidate[] | null>(null);
  const [resolveError, setResolveError] = useState<string | null>(null);
  const [job, setJob] = useState<IngestStatus | null>(null);
  const [starting, setStarting] = useState<string | null>(null);
  const [corpus, setCorpus] = useState<CorpusResponse | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refreshCorpus = () =>
    fetchCorpus().then(setCorpus).catch(() => setCorpus(null));

  useEffect(() => {
    refreshCorpus();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  async function doResolve(q: string) {
    const query = q.trim();
    if (!query || resolving) return;
    setResolving(true);
    setResolveError(null);
    setCandidates(null);
    setJob(null);
    try {
      const { candidates } = await resolveCompany(query);
      setCandidates(candidates);
    } catch (err) {
      setResolveError(
        err instanceof Error
          ? err.message
          : "Resolve failed — is the API running on port 8000?",
      );
    } finally {
      setResolving(false);
    }
  }

  async function doIngest(ticker: string) {
    if (starting) return;
    setStarting(ticker);
    try {
      const { job_id } = await startIngest(ticker);
      const tick = async () => {
        try {
          const status = await fetchIngestStatus(job_id);
          setJob(status);
          if (status.stage === "ready" || status.stage === "error") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            if (status.stage === "ready") refreshCorpus();
          }
        } catch {
          /* transient poll failure — keep polling */
        }
      };
      await tick();
      pollRef.current = setInterval(tick, 1200);
    } catch (err) {
      setJob({
        job_id: "",
        ticker,
        stage: "error",
        stages: [],
        result: null,
        error:
          err instanceof Error ? err.message : "Could not start the ingest job.",
      });
    } finally {
      setStarting(null);
    }
  }

  const sessionCompanies = corpus?.session_companies ?? [];
  const jobRunning = job !== null && job.stage !== "ready" && job.stage !== "error";
  const result = job?.stage === "ready" ? job.result : null;

  return (
    <div className="py-8 lg:py-10">
      <Reveal>
        <h1 className="font-display text-4xl font-medium tracking-tight text-balance sm:text-[44px] sm:leading-[1.05]">
          Add a Company<span className="text-accent-text">.</span>
        </h1>
        <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-muted-foreground">
          Point the pipeline at any SEC-listed company: it resolves the
          ticker, fetches the latest 10-K from EDGAR, slices it by Item with
          the same caps as the bundled corpus, chunks it both ways, embeds and
          indexes it, and builds its figures table — live. The company joins
          this session&apos;s corpus, and the chat can answer from it
          immediately. The bundled snapshot — and every measurement on the Lab
          and Eval pages — stays exactly as it was.
        </p>
      </Reveal>

      <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0 space-y-6">
          {/* ——— Resolve ——— */}
          <Reveal delay={0.08}>
            <Card>
              <CardHeader>
                <CardTitle>Find a company</CardTitle>
                <p className="text-[12.5px] text-muted-foreground">
                  Search the SEC company map by name or ticker.
                </p>
              </CardHeader>
              <CardContent className="space-y-4">
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    doResolve(query);
                  }}
                  className="flex items-end gap-2"
                >
                  <div className="flex-1">
                    <label htmlFor="company-query" className="sr-only">
                      Company name or ticker
                    </label>
                    <input
                      id="company-query"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="e.g. Coca-Cola, KO, Tesla…"
                      className="min-h-[44px] w-full rounded-md border border-border bg-transparent px-3 py-2 text-[14.5px] outline-none placeholder:text-muted-foreground/70"
                    />
                  </div>
                  <Button type="submit" disabled={resolving || !query.trim()}>
                    {resolving ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Search className="h-4 w-4" />
                    )}
                    Find company
                  </Button>
                </form>
                <div className="flex flex-wrap gap-1.5">
                  {QUICK_PICKS.map((p) => (
                    <button
                      key={p.q}
                      type="button"
                      disabled={resolving}
                      onClick={() => {
                        setQuery(p.q);
                        doResolve(p.q);
                      }}
                      className="min-h-[36px] cursor-pointer rounded-md border border-border px-3 py-1.5 text-[12.5px] transition-colors duration-200 hover:bg-accent disabled:opacity-50"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>

                {resolveError && (
                  <p className="flex items-start gap-2 text-sm text-loss-text">
                    <CircleAlert className="mt-0.5 h-4 w-4 shrink-0" />
                    {resolveError}
                  </p>
                )}
                {candidates !== null && candidates.length === 0 && (
                  <p className="rounded-lg border border-loss/30 bg-loss/5 px-4 py-3 text-[13.5px] leading-relaxed">
                    No SEC-listed company matches “{query}”. Resolution is
                    honest: an unknown name stays unknown — try a ticker
                    symbol or the company&apos;s legal name.
                  </p>
                )}
                {candidates !== null && candidates.length > 0 && (
                  <div className="space-y-2">
                    {candidates.map((c) => (
                      <div
                        key={c.ticker}
                        className="flex flex-wrap items-center gap-3 rounded-lg border border-border px-4 py-3"
                      >
                        <Building2 className="h-4 w-4 shrink-0 text-muted-foreground" />
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-[14px] font-medium">{c.name}</p>
                          <p className="tnum font-mono text-[11.5px] text-muted-foreground">
                            CIK {c.cik}
                          </p>
                        </div>
                        <Badge variant="gold">{c.ticker}</Badge>
                        <Button
                          size="sm"
                          disabled={starting !== null || jobRunning}
                          onClick={() => doIngest(c.ticker)}
                        >
                          {starting === c.ticker ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : null}
                          Fetch latest 10-K
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </Reveal>

          {/* ——— Pipeline ——— */}
          {job && (
            <Reveal delay={0.05}>
              <Card>
                <CardHeader>
                  <CardTitle>
                    Ingesting {job.ticker}
                    {job.stage === "ready" && " — done"}
                    {job.stage === "error" && " — failed"}
                  </CardTitle>
                  <p className="text-[12.5px] text-muted-foreground">
                    The same pipeline that built the bundled corpus, run live
                    for one company.
                  </p>
                </CardHeader>
                <CardContent className="space-y-4">
                  <ol className="space-y-3">
                    {job.stages.map((s) => (
                      <li key={s.name} className="flex items-start gap-3">
                        <StageIcon status={s.status} />
                        <div className="min-w-0">
                          <p
                            className={cn(
                              "text-[13.5px] font-medium",
                              s.status === "pending" && "text-muted-foreground",
                              s.status === "active" && "text-accent-text",
                            )}
                          >
                            {STAGE_LABELS[s.name] ?? s.name}
                          </p>
                          <p className="text-[12.5px] leading-relaxed text-muted-foreground">
                            {s.detail}
                          </p>
                        </div>
                      </li>
                    ))}
                  </ol>

                  {job.stage === "error" && (
                    <p className="rounded-lg border border-loss/30 bg-loss/5 px-4 py-3 text-[13.5px] leading-relaxed">
                      <span className="font-medium text-loss-text">Ingest failed. </span>
                      {job.error ?? "Unknown error."} Nothing was added to the
                      corpus — the registry only registers a company when
                      every stage completes.
                    </p>
                  )}

                  {result && (
                    <div className="space-y-4 border-t border-border pt-4">
                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                        {[
                          { k: "Filing", v: `${result.ticker} 10-K FY${result.fiscal_year}` },
                          { k: "Accession", v: result.accession },
                          { k: "Items sliced", v: result.items.join(", ") || "—" },
                          { k: "Characters", v: result.chars.toLocaleString() },
                          {
                            k: "Chunks",
                            v: `${result.chunks.fixed} fixed · ${result.chunks.section} section`,
                          },
                          { k: "Figures rows", v: String(result.figures_rows) },
                        ].map((s) => (
                          <div key={s.k} className="rounded-lg border border-border px-3 py-2.5">
                            <p className="text-[11px] tracking-wide text-muted-foreground">
                              {s.k}
                            </p>
                            <p className="tnum mt-0.5 font-mono text-[12.5px]">{s.v}</p>
                          </div>
                        ))}
                      </div>
                      <p className="text-[12.5px] text-muted-foreground">
                        Source:{" "}
                        {result.source === "precached"
                          ? "precached snapshot of the real EDGAR pipeline output (zero network for this demo run)"
                          : result.source === "live"
                            ? "fetched live from SEC EDGAR just now"
                            : "bundled corpus"}
                        {" "}· filed {result.filed} · period ended {result.report_date}
                      </p>
                      <Link
                        href="/"
                        className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors duration-200 hover:bg-accent-bright"
                      >
                        Ask it in Chat
                        <ArrowRight className="h-4 w-4" />
                      </Link>
                    </div>
                  )}
                </CardContent>
              </Card>
            </Reveal>
          )}
        </div>

        {/* ——— Session companies ——— */}
        <aside className="lg:sticky lg:top-24 lg:self-start">
          <Reveal delay={0.14}>
            <Card>
              <CardHeader>
                <CardTitle>In this session</CardTitle>
                <p className="text-[12.5px] text-muted-foreground">
                  Companies ingested live, on top of the bundled AAPL · MSFT ·
                  NVDA snapshot. They persist on the server, so a page reload
                  keeps them.
                </p>
              </CardHeader>
              <CardContent className="space-y-3">
                {sessionCompanies.length === 0 ? (
                  <p className="text-[13px] text-muted-foreground">
                    None yet — the chat currently answers from the bundled
                    snapshot only. Try the Tesla quick pick: it runs the full
                    pipeline from a precached snapshot, no network needed.
                  </p>
                ) : (
                  sessionCompanies.map((c) => (
                    <div key={c.ticker} className="flex items-baseline gap-2 text-[13px]">
                      <span className="font-medium">{c.ticker}</span>
                      <span className="min-w-0 flex-1 truncate text-muted-foreground">
                        {c.name}
                      </span>
                      <Badge variant="gold" className="shrink-0">
                        {c.source}
                      </Badge>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
          </Reveal>
        </aside>
      </div>
    </div>
  );
}
