"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowUp, CircleAlert, HelpCircle, Landmark, MessageSquarePlus, Sparkles, X } from "lucide-react";
import {
  fetchCorpus,
  postChat,
  type ChatResponse,
  type CorpusResponse,
} from "@/lib/api";
import { useChatSession, type Turn } from "@/components/chat-session";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EvidenceCard } from "@/components/evidence-card";
import { MarketStrip } from "@/components/market-strip";
import { CompanySnapshot } from "@/components/company-snapshot";
import { PendingTrail, StepTrail } from "@/components/step-trail";
import { Reveal } from "@/components/reveal";
import { cn } from "@/lib/utils";

const BUNDLED_SHORT_NAMES: Record<string, string> = {
  AAPL: "Apple",
  MSFT: "Microsoft",
  NVDA: "NVIDIA",
};

function suggestions(teslaIngested: boolean) {
  return [
    { label: "Apple's FY2024 net sales", q: "What was Apple's total net sales in fiscal 2024?" },
    { label: "Microsoft's risk factors", q: "What risk factors does Microsoft disclose about competition and cybersecurity threats?" },
    { label: "Apple vs NVIDIA revenue", q: "Which company reported higher total revenue in its most recent fiscal year, Apple or NVIDIA, and what were the figures?" },
    {
      label: teslaIngested
        ? "Tesla's FY2024 revenue — ingested this session"
        : "Tesla — outside the corpus",
      q: "What was Tesla's revenue in fiscal 2024?",
    },
    { label: "“What are the risks?” — ambiguous", q: "What are the risks?" },
  ];
}

function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string; hint: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div>
      <p className="mb-2 text-[12px] font-medium tracking-wide text-muted-foreground">
        {label}
      </p>
      <div className="flex flex-col gap-1.5" role="group" aria-label={label}>
        {options.map((opt) => {
          const active = opt.value === value;
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => onChange(opt.value)}
              aria-pressed={active}
              className={cn(
                "min-h-[44px] cursor-pointer rounded-md border px-3 py-2 text-left transition-colors duration-200",
                active
                  ? "border-primary/50 bg-primary/10"
                  : "border-border bg-transparent hover:bg-accent",
              )}
            >
              <span
                className={cn(
                  "block text-[13px] font-medium",
                  active ? "text-accent-text" : "text-foreground",
                )}
              >
                {opt.label}
              </span>
              <span className="block text-[11.5px] text-muted-foreground">
                {opt.hint}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function AnswerBlock({ result }: { result: ChatResponse }) {
  if (result.refused) {
    return (
      <div className="rounded-lg border border-loss/30 bg-loss/5 px-4 py-3.5">
        <p className="mb-1.5 flex items-center gap-2 text-[13px] font-medium text-loss-text">
          <CircleAlert className="h-4 w-4" /> Not in the filings
        </p>
        <p className="text-[14px] leading-relaxed text-foreground/90">{result.answer}</p>
      </div>
    );
  }
  if (result.clarification) {
    return (
      <div className="rounded-lg border border-primary/30 bg-primary/5 px-4 py-3.5">
        <p className="mb-1.5 flex items-center gap-2 text-[13px] font-medium text-accent-text">
          <HelpCircle className="h-4 w-4" /> One clarification
        </p>
        <p className="text-[14px] leading-relaxed text-foreground/90">{result.answer}</p>
      </div>
    );
  }
  return (
    <div className="space-y-3">
      {result.answer.split(/\n\n+/).map((para, i) => (
        <p key={i} className="text-[14.5px] leading-relaxed text-foreground/92">
          {para}
        </p>
      ))}
    </div>
  );
}

export default function ChatPage() {
  // Transcript + conversation context live in the layout-level session
  // store, so they survive in-app navigation; "New chat" resets them.
  const {
    turns, setTurns,
    input, setInput,
    strategy, setStrategy,
    mode, setMode,
    corpusChoice, setCorpusChoice,
    context, setContext,
  } = useChatSession();
  const [pendingStage, setPendingStage] = useState(0);
  const [corpus, setCorpus] = useState<CorpusResponse | null>(null);
  const [corpusError, setCorpusError] = useState(false);
  const [focusTicker, setFocusTicker] = useState("AAPL");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const idRef = useRef(0);

  // Corpus stats: retry transient failures (a cold API can take a few
  // seconds to wake) and never blank out stats we already have — the
  // "unavailable" notice is reserved for a genuine, repeated failure.
  const refreshCorpus = async () => {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        setCorpus(await fetchCorpus());
        setCorpusError(false);
        return;
      } catch {
        if (attempt < 2) {
          await new Promise((r) => setTimeout(r, 1200 * (attempt + 1)));
        }
      }
    }
    setCorpusError(true);
  };

  useEffect(() => {
    refreshCorpus();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  useEffect(() => {
    if (!busy) return;
    setPendingStage(0);
    const t = setInterval(
      () => setPendingStage((s) => Math.min(s + 1, 4)),
      1300,
    );
    return () => clearInterval(t);
  }, [busy]);

  async function ask(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    const id = ++idRef.current;
    setTurns((t) => [...t, { id, question: q, status: "pending" }]);
    setInput("");
    setBusy(true);
    try {
      const result = await postChat(q, strategy, mode, corpusChoice, context);
      if (result.resolved?.ticker) setContext(result.resolved);
      setTurns((t) =>
        t.map((turn) => (turn.id === id ? { ...turn, status: "done", result } : turn)),
      );
    } catch (err) {
      setTurns((t) =>
        t.map((turn) =>
          turn.id === id
            ? {
                ...turn,
                status: "error",
                error:
                  err instanceof Error
                    ? err.message
                    : "The API did not respond — is `uvicorn src.api:app --port 8000` running?",
              }
            : turn,
        ),
      );
    } finally {
      setBusy(false);
      refreshCorpus();
    }
  }

  /** Explicit reset: clears this conversation (transcript + follow-up
   *  context). The corpus — bundled and session-ingested companies —
   *  is untouched; it is the session, not the conversation. */
  function newChat() {
    if (busy) return;
    setTurns([]);
    setContext(null);
    setInput("");
  }

  function contextName(ticker: string): string {
    const session = (corpus?.session_companies ?? []).find(
      (c) => c.ticker === ticker,
    );
    if (session) {
      return session.name.replace(/,?\s+(Inc\.?|Corp\.?|Corporation|Company|Ltd\.?)$/i, "");
    }
    return BUNDLED_SHORT_NAMES[ticker] ?? ticker;
  }

  const sessionCompanies = corpus?.session_companies ?? [];
  const teslaIngested = sessionCompanies.some((c) => c.ticker === "TSLA");

  // Snapshot focus follows the conversation: the company behind the
  // most recent cited answer (citation doc labels start with the
  // ticker), or a corpus-panel click.
  useEffect(() => {
    if (!corpus) return;
    const known = new Set(corpus.documents.map((d) => d.ticker));
    const last = [...turns]
      .reverse()
      .find(
        (t) =>
          t.status === "done" &&
          t.result &&
          !t.result.refused &&
          t.result.citations.length > 0,
      );
    if (!last?.result) return;
    const candidate = last.result.citations[0].doc
      .split(/[\s·]/)[0]
      ?.toUpperCase();
    if (candidate && known.has(candidate)) setFocusTicker(candidate);
  }, [turns, corpus]);

  return (
    <div className="py-8 lg:py-10">
      <Reveal>
        <h1 className="font-display text-4xl font-medium tracking-tight text-balance sm:text-[44px] sm:leading-[1.05]">
          Ask the filings<span className="text-accent-text">.</span>
        </h1>
        <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-muted-foreground">
          Every answer is retrieved from real 10-K filings, graded before it is
          written, and verified figure-by-figure before you see it. When the
          filings don't contain the answer, this system says so — and shows its
          working either way.
        </p>
      </Reveal>

      <div className="mt-6">
        <MarketStrip />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        {/* ——— Conversation column: transcript scrolls, composer sticks
            to the bottom of the viewport (ChatGPT-style) ——— */}
        <div className="flex min-w-0 flex-col">
          <div className="flex-1 space-y-8">
            {turns.length === 0 && (
              <Reveal delay={0.15}>
                <div className="rounded-xl border border-dashed border-border px-5 py-8 text-center">
                  <Landmark className="mx-auto h-5 w-5 text-muted-foreground" />
                  <p className="mt-3 text-sm text-muted-foreground">
                    No questions yet. Try a suggestion from the panel — including
                    the two designed to fail honestly.
                  </p>
                </div>
              </Reveal>
            )}
            {turns.map((turn) => (
              <article key={turn.id} className="space-y-4">
                <div className="flex justify-end">
                  <div className="max-w-[85%] rounded-xl rounded-br-sm border border-border bg-secondary px-4 py-3">
                    <p className="text-[14.5px] leading-relaxed">{turn.question}</p>
                  </div>
                </div>
                {turn.status === "done" && turn.result?.understood_as && (
                  <p className="text-right text-[12.5px] text-muted-foreground">
                    Understood as:{" "}
                    <span className="text-foreground/80">
                      &ldquo;{turn.result.understood_as}&rdquo;
                    </span>
                  </p>
                )}
                <div className="rounded-xl border border-border bg-card px-5 py-5">
                  {turn.status === "pending" && <PendingTrail stage={pendingStage} />}
                  {turn.status === "error" && (
                    <p className="flex items-start gap-2 text-sm text-loss-text">
                      <CircleAlert className="mt-0.5 h-4 w-4 shrink-0" />
                      {turn.error}
                    </p>
                  )}
                  {turn.status === "done" && turn.result && (
                    <div className="space-y-5">
                      <StepTrail trace={turn.result.trace} />
                      <div className="border-t border-border pt-4">
                        <AnswerBlock result={turn.result} />
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant={turn.result.mode === "llm" ? "gold" : "default"}>
                          <Sparkles className="h-3 w-3" />
                          {turn.result.mode === "llm"
                            ? "LLM generation"
                            : "Extractive grounded mode"}
                        </Badge>
                        <Badge variant="muted">
                          <span className="tnum">
                            {(turn.result.latency_ms / 1000).toFixed(1)}s end-to-end
                          </span>
                        </Badge>
                        <Badge variant="muted">
                          {turn.result.strategy === "section"
                            ? "Section-aware chunks"
                            : "Fixed-size chunks"}{" "}
                          · {turn.result.retrieval_mode}
                        </Badge>
                      </div>
                      {turn.result.citations.length > 0 && (
                        <div className="space-y-2">
                          <p className="text-[12px] font-medium tracking-wide text-muted-foreground">
                            Evidence
                          </p>
                          {turn.result.citations.map((c, i) => (
                            <EvidenceCard key={i} citation={c} index={i} />
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </article>
            ))}
            <div ref={bottomRef} className="scroll-mb-56" />
          </div>

          {/* ——— Sticky bottom bar: context chip, New chat, composer ——— */}
          <div className="sticky bottom-0 z-10 -mx-1 bg-gradient-to-t from-background via-background/97 to-transparent px-1 pt-5 pb-3">
            {(turns.length > 0 || context?.ticker) && (
              <div className="mb-2 flex flex-wrap items-center gap-2">
                {context?.ticker && (
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-primary/40 bg-primary/10 px-2.5 py-1 text-[12px] text-accent-text">
                    Context: {contextName(context.ticker)}
                    {context.fiscal_year ? ` · FY${context.fiscal_year}` : ""}
                    <button
                      type="button"
                      aria-label="Clear conversation context"
                      title="Clear context — keeps the transcript"
                      onClick={() => setContext(null)}
                      className="cursor-pointer rounded-full p-0.5 transition-colors hover:bg-primary/20"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </span>
                )}
                <span className="ml-auto flex items-center gap-3">
                  <span className="hidden text-[11.5px] text-muted-foreground md:block">
                    New chat clears this conversation — your corpus stays
                  </span>
                  <button
                    type="button"
                    onClick={newChat}
                    disabled={busy}
                    className="inline-flex min-h-[32px] cursor-pointer items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-[12.5px] text-muted-foreground transition-colors duration-200 hover:bg-accent hover:text-foreground disabled:opacity-50"
                  >
                    <MessageSquarePlus className="h-3.5 w-3.5" />
                    New chat
                  </button>
                </span>
              </div>
            )}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                ask(input);
              }}
              className="rounded-xl border border-border bg-card p-3"
            >
              <label htmlFor="question" className="sr-only">
                Ask a question about the filings
              </label>
              <div className="flex items-end gap-2">
                <textarea
                  id="question"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      ask(input);
                    }
                  }}
                  rows={2}
                  placeholder="e.g. What was Apple's total net sales in fiscal 2024?"
                  className="max-h-32 min-h-[44px] flex-1 resize-none bg-transparent px-2 py-2 text-[14.5px] leading-relaxed outline-none placeholder:text-muted-foreground/70"
                />
                <Button type="submit" disabled={busy || !input.trim()} className="shrink-0">
                  <ArrowUp className="h-4 w-4" />
                  Ask
                </Button>
              </div>
            </form>
          </div>
        </div>

        {/* ——— Controls + corpus sidebar ——— */}
        <aside className="space-y-6 lg:sticky lg:top-24 lg:self-start">
          <Reveal delay={0.12}>
            <Card>
              <CardHeader>
                <CardTitle>Retrieval</CardTitle>
                <p className="text-[12.5px] text-muted-foreground">
                  Flip these mid-conversation — the next answer runs through the
                  configuration you pick. The Chunking Lab shows what each costs
                  and buys.
                </p>
              </CardHeader>
              <CardContent className="space-y-5">
                <Segmented
                  label="Chunking strategy"
                  value={strategy}
                  onChange={setStrategy}
                  options={[
                    {
                      value: "section",
                      label: "Section-aware",
                      hint: "Split on Item headings, ~4,000-char packs — never crosses a section",
                    },
                    {
                      value: "fixed",
                      label: "Fixed-size",
                      hint: "Recursive splitter, 1,000 chars / 200 overlap",
                    },
                  ]}
                />
                <Segmented
                  label="Retrieval mode"
                  value={mode}
                  onChange={setMode}
                  options={[
                    { value: "hybrid+rerank", label: "Hybrid + rerank", hint: "BM25 + vector (RRF), cross-encoder rerank" },
                    { value: "hybrid", label: "Hybrid", hint: "BM25 + vector fused with RRF (k=60)" },
                    { value: "vector", label: "Vector only", hint: "FAISS dense similarity, bge-small-en-v1.5" },
                  ]}
                />
              </CardContent>
            </Card>
          </Reveal>

          <Reveal delay={0.18}>
            <Card>
              <CardHeader>
                <CardTitle>Try asking</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-1.5">
                {suggestions(teslaIngested).map((s) => (
                  <button
                    key={s.q}
                    type="button"
                    disabled={busy}
                    onClick={() => ask(s.q)}
                    className="min-h-[44px] cursor-pointer rounded-md border border-border px-3 py-2 text-left text-[13px] transition-colors duration-200 hover:bg-accent disabled:opacity-50"
                  >
                    {s.label}
                  </button>
                ))}
              </CardContent>
            </Card>
          </Reveal>

          <Reveal delay={0.24}>
            <Card>
              <CardHeader>
                <CardTitle>Corpus</CardTitle>
                <p className="text-[12.5px] text-muted-foreground">
                  SEC EDGAR 10-K excerpts, cleaned and section-mapped. The
                  bundled snapshot is frozen; companies added via{" "}
                  <Link href="/ingest" className="text-accent-text underline underline-offset-2">
                    Add a Company
                  </Link>{" "}
                  join the active corpus for this session.
                </p>
              </CardHeader>
              <CardContent className="space-y-3">
                {corpus ? (
                  <>
                    <Segmented
                      label="Search scope"
                      value={corpusChoice}
                      onChange={setCorpusChoice}
                      options={[
                        {
                          value: "active",
                          label: "Active corpus",
                          hint:
                            sessionCompanies.length > 0
                              ? `Bundled snapshot + ${sessionCompanies
                                  .map((c) => c.ticker)
                                  .join(", ")} (session)`
                              : "Bundled snapshot (no session companies yet)",
                        },
                        {
                          value: "bundled",
                          label: "Bundled only",
                          hint: "AAPL · MSFT · NVDA snapshot — the measured corpus",
                        },
                        ...sessionCompanies.map((c) => ({
                          value: c.ticker,
                          label: `${c.name} only`,
                          hint: `Session company · 10-K FY${c.fiscal_year}`,
                        })),
                      ]}
                    />
                    {corpus.documents.map((d) => (
                      <button
                        key={d.doc_id}
                        type="button"
                        onClick={() => setFocusTicker(d.ticker)}
                        title={`Show the ${d.ticker} snapshot`}
                        className={cn(
                          "flex w-full cursor-pointer items-baseline gap-2 rounded px-1 py-0.5 text-left text-[13px] transition-colors duration-200 hover:bg-accent",
                          focusTicker === d.ticker && "bg-primary/10",
                        )}
                      >
                        <span className="font-medium">{d.ticker}</span>
                        <span className="min-w-0 flex-1 truncate text-muted-foreground">
                          {d.label.replace(`${d.ticker} `, "")}
                        </span>
                        {d.source !== "bundled" && (
                          <Badge variant="gold" className="shrink-0">
                            session
                          </Badge>
                        )}
                        <span className="tnum font-mono text-[11.5px] text-muted-foreground">
                          {(d.chars / 1000).toFixed(0)}k chars
                        </span>
                      </button>
                    ))}
                    <div className="space-y-1 border-t border-border pt-3 text-[12.5px] text-muted-foreground">
                      <p className="flex justify-between">
                        <span>Fixed-size chunks (bundled)</span>
                        <span className="tnum font-mono">{corpus.chunks.fixed}</span>
                      </p>
                      <p className="flex justify-between">
                        <span>Section-aware chunks (bundled)</span>
                        <span className="tnum font-mono">{corpus.chunks.section}</span>
                      </p>
                      {sessionCompanies.map((c) => (
                        <p key={c.ticker} className="flex justify-between">
                          <span>{c.ticker} chunks (session)</span>
                          <span className="tnum font-mono">
                            {c.chunks.fixed} fixed · {c.chunks.section} section
                          </span>
                        </p>
                      ))}
                      <p className="flex justify-between">
                        <span>Generation</span>
                        <span>
                          {corpus.generation_mode === "llm" ? "LLM (API key set)" : "Extractive (no key)"}
                        </span>
                      </p>
                    </div>
                  </>
                ) : corpusError ? (
                  <p className="text-[13px] text-muted-foreground">
                    Corpus stats unavailable right now — chat and retrieval
                    still work, and this panel retries on the next question.
                  </p>
                ) : (
                  <p className="text-[13px] text-muted-foreground">
                    Loading corpus stats…
                  </p>
                )}
              </CardContent>
            </Card>
          </Reveal>

          <Reveal delay={0.3}>
            <CompanySnapshot ticker={focusTicker} />
          </Reveal>
        </aside>
      </div>
    </div>
  );
}
