"use client";

import { useEffect, useState } from "react";
import { FileText } from "lucide-react";
import {
  fetchFilingFigures,
  fetchMarketHistory,
  fetchMarketQuote,
  type FilingFigures,
  type MarketHistory,
  type MarketQuote,
} from "@/lib/api";
import { fmtMoney, fmtPct, fmtPrice, fmtVolume, sourceLabel } from "@/lib/format";
import { PriceAreaChart } from "@/components/price-chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const RANGES = [
  { value: "1mo", label: "1M" },
  { value: "6mo", label: "6M" },
  { value: "1y", label: "1Y" },
] as const;

function Shimmer() {
  return (
    <div className="animate-pulse space-y-3" aria-hidden="true">
      <div className="h-7 w-32 rounded bg-secondary" />
      <div className="h-[120px] rounded bg-secondary" />
      <div className="grid grid-cols-2 gap-2">
        <div className="h-10 rounded bg-secondary" />
        <div className="h-10 rounded bg-secondary" />
      </div>
    </div>
  );
}

function Stat({ k, v }: { k: string; v: string | null }) {
  if (v === null) return null;
  return (
    <div className="rounded-lg border border-border px-3 py-2">
      <p className="text-[11px] tracking-wide text-muted-foreground">{k}</p>
      <p className="tnum mt-0.5 text-[13px] font-medium">{v}</p>
    </div>
  );
}

/**
 * Company snapshot — the market's view of the company in focus next to
 * the filing's view of it: delayed quote + price chart + market stats
 * above, and the corpus figures table (revenue / net income by fiscal
 * year) in a clearly separated, source-labelled block below.
 */
export function CompanySnapshot({ ticker }: { ticker: string }) {
  const [quote, setQuote] = useState<MarketQuote | null>(null);
  const [history, setHistory] = useState<MarketHistory | null>(null);
  const [figures, setFigures] = useState<FilingFigures | null>(null);
  const [range, setRange] = useState<string>("6mo");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    Promise.allSettled([
      fetchMarketQuote(ticker),
      fetchFilingFigures(ticker),
    ]).then(([q, f]) => {
      if (!alive) return;
      setQuote(q.status === "fulfilled" && q.value.available ? q.value : null);
      setFigures(f.status === "fulfilled" ? f.value : null);
      setLoading(false);
    });
    return () => {
      alive = false;
    };
  }, [ticker]);

  useEffect(() => {
    let alive = true;
    fetchMarketHistory(ticker, range)
      .then((h) => {
        if (alive) setHistory(h.available ? h : null);
      })
      .catch(() => {
        if (alive) setHistory(null);
      });
    return () => {
      alive = false;
    };
  }, [ticker, range]);

  const latest = figures?.rows[0];
  const prior = figures?.rows[1];
  const revDelta =
    latest?.revenue && prior?.revenue
      ? ((latest.revenue - prior.revenue) / prior.revenue) * 100
      : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Company snapshot</CardTitle>
        <p className="text-[12.5px] text-muted-foreground">
          The market&apos;s numbers beside the filing&apos;s numbers — each
          labelled with where it came from.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading ? (
          <Shimmer />
        ) : quote ? (
          <>
            <div>
              <p className="text-[13px] text-muted-foreground">
                {quote.name} · {quote.ticker}
              </p>
              <p className="tnum mt-0.5 flex flex-wrap items-baseline gap-2">
                <span className="text-[26px] font-medium leading-none">
                  ${fmtPrice(quote.price)}
                </span>
                <span
                  className={cn(
                    "text-[13px] font-medium",
                    quote.change_pct === null
                      ? "text-muted-foreground"
                      : quote.change_pct >= 0
                        ? "text-gain-text"
                        : "text-loss-text",
                  )}
                >
                  {quote.change !== null && quote.change !== undefined
                    ? `${quote.change >= 0 ? "+" : ""}${quote.change.toFixed(2)} `
                    : ""}
                  ({fmtPct(quote.change_pct)}) today
                </span>
              </p>
            </div>

            <div>
              <div className="mb-2 flex items-center justify-between">
                <p className="text-[11.5px] tracking-wide text-muted-foreground">
                  Price history
                </p>
                <div className="flex gap-1" role="group" aria-label="Chart range">
                  {RANGES.map((r) => (
                    <button
                      key={r.value}
                      type="button"
                      onClick={() => setRange(r.value)}
                      aria-pressed={range === r.value}
                      className={cn(
                        "min-h-[28px] cursor-pointer rounded px-2 text-[11.5px] font-medium transition-colors duration-200",
                        range === r.value
                          ? "bg-primary/15 text-accent-text"
                          : "text-muted-foreground hover:bg-accent",
                      )}
                    >
                      {r.label}
                    </button>
                  ))}
                </div>
              </div>
              <PriceAreaChart
                values={history ? history.points.map((p) => p.close) : []}
              />
              <p className="mt-1.5 text-[11px] text-muted-foreground">
                Delayed · {sourceLabel(quote.source)}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <Stat k="Market cap" v={fmtMoney(quote.market_cap)} />
              <Stat
                k="P/E (trailing)"
                v={quote.pe_ratio ? quote.pe_ratio.toFixed(1) : null}
              />
              <Stat
                k="52-week range"
                v={
                  quote.low_52w && quote.high_52w
                    ? `${fmtPrice(quote.low_52w)} – ${fmtPrice(quote.high_52w)}`
                    : null
                }
              />
              <Stat
                k="Day range"
                v={
                  quote.day_low && quote.day_high
                    ? `${fmtPrice(quote.day_low)} – ${fmtPrice(quote.day_high)}`
                    : null
                }
              />
              <Stat k="Volume" v={fmtVolume(quote.volume)} />
              <Stat k="Prev close" v={quote.prev_close ? `$${fmtPrice(quote.prev_close)}` : null} />
            </div>
          </>
        ) : (
          <p className="text-[13px] text-muted-foreground">
            Market data unavailable for {ticker} right now — the filings
            below are unaffected.
          </p>
        )}

        {figures?.available && latest && (
          <div className="space-y-2.5 border-t border-border pt-4">
            <p className="flex items-center gap-1.5 text-[12px] font-medium tracking-wide text-accent-text">
              <FileText className="h-3.5 w-3.5" />
              From the filing — our corpus
            </p>
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-lg border border-primary/25 bg-primary/5 px-3 py-2">
                <p className="text-[11px] tracking-wide text-muted-foreground">
                  Revenue · FY{latest.fiscal_year}
                </p>
                <p className="tnum mt-0.5 text-[14px] font-medium">
                  {fmtMoney(latest.revenue)}
                  {revDelta !== null && (
                    <span
                      className={cn(
                        "ml-1.5 text-[11.5px]",
                        revDelta >= 0 ? "text-gain-text" : "text-loss-text",
                      )}
                    >
                      {fmtPct(revDelta)} YoY
                    </span>
                  )}
                </p>
              </div>
              <div className="rounded-lg border border-primary/25 bg-primary/5 px-3 py-2">
                <p className="text-[11px] tracking-wide text-muted-foreground">
                  Net income · FY{latest.fiscal_year}
                </p>
                <p className="tnum mt-0.5 text-[14px] font-medium">
                  {fmtMoney(latest.net_income)}
                </p>
              </div>
            </div>
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              {figures.source} — the same figures the chat cites. Companies
              appear here once their company-facts table is in the corpus.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
