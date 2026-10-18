/** Display formatting for market + filing figures. */

export function fmtPrice(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return v.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

export function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

/** $5.01T / $391.0B / $97.7M — for market caps and filing figures. */
export function fmtMoney(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  const abs = Math.abs(v);
  const fmt = (n: number, suffix: string) =>
    `$${n.toLocaleString("en-US", { maximumFractionDigits: 2 })}${suffix}`;
  if (abs >= 1e12) return fmt(v / 1e12, "T");
  if (abs >= 1e9) return fmt(v / 1e9, "B");
  if (abs >= 1e6) return fmt(v / 1e6, "M");
  return `$${v.toLocaleString("en-US")}`;
}

export function fmtVolume(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return String(v);
}

/** "as of 4:02 PM ET" from an ISO timestamp. */
export function fmtAsOfET(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString("en-US", {
    timeZone: "America/New_York",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function sourceLabel(source: string | null | undefined): string {
  if (source === "yahoo") return "Yahoo Finance";
  if (source === "stooq") return "Stooq";
  return "market data";
}
