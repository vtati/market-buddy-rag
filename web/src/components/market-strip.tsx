"use client";

import { useEffect, useState } from "react";
import { TrendingDown, TrendingUp } from "lucide-react";
import {
  fetchMarketOverview,
  type MarketOverview,
  type MarketQuote,
} from "@/lib/api";
import { fmtAsOfET, fmtPct, fmtPrice, sourceLabel } from "@/lib/format";
import { Sparkline } from "@/components/price-chart";
import { cn } from "@/lib/utils";

function StripItem({ q, label }: { q: MarketQuote; label?: string }) {
  const positive = q.change_pct === null ? null : q.change_pct >= 0;
  return (
    <div className="flex min-w-[168px] items-center gap-3 px-4 py-2.5">
      <div className="min-w-0 flex-1">
        <p className="truncate text-[11.5px] font-medium tracking-wide text-muted-foreground">
          {label ?? q.name}
        </p>
        <p className="tnum text-[14px] font-medium leading-tight">
          {fmtPrice(q.price)}
          <span
            className={cn(
              "ml-2 inline-flex items-center gap-0.5 text-[12px] font-medium",
              positive === null
                ? "text-muted-foreground"
                : positive
                  ? "text-gain-text"
                  : "text-loss-text",
            )}
          >
            {positive !== null &&
              (positive ? (
                <TrendingUp className="h-3 w-3" />
              ) : (
                <TrendingDown className="h-3 w-3" />
              ))}
            {fmtPct(q.change_pct)}
          </span>
        </p>
      </div>
      <Sparkline values={q.spark ?? []} positive={positive} />
    </div>
  );
}

/**
 * Market strip — the tape above the conversation: the three indices
 * plus every active-corpus company, refreshed every 60 seconds. If
 * market data is unavailable the strip renders nothing at all: a
 * silent absence beats a broken shell.
 */
export function MarketStrip() {
  const [overview, setOverview] = useState<MarketOverview | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      fetchMarketOverview()
        .then((o) => {
          if (alive) setOverview(o.available ? o : null);
        })
        .catch(() => {
          if (alive) setOverview(null);
        });
    load();
    const t = setInterval(load, 60_000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  if (!overview) return null;
  const items = [...overview.indices, ...overview.companies].filter(
    (q) => q.available,
  );
  if (items.length === 0) return null;

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-card">
      <div className="flex flex-wrap divide-x divide-border">
        {items.map((q) => (
          <StripItem key={q.ticker} q={q} />
        ))}
      </div>
      <p className="border-t border-border px-4 py-1.5 text-[11px] text-muted-foreground">
        Delayed quotes · {sourceLabel(overview.source)} · as of{" "}
        {fmtAsOfET(overview.as_of)} ET — market context only; answers come
        from the filings.
      </p>
    </div>
  );
}
