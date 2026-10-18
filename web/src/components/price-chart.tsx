"use client";

/**
 * Hand-rolled SVG price charts in the house style — gold stroke with a
 * soft gold fill for price, green/red semantics for day change. No
 * chart library: these are paths, not a dependency.
 */

function pathFor(values: number[], width: number, height: number, pad = 2) {
  if (values.length === 0) return { line: "", area: "" };
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const stepX = values.length > 1 ? (width - pad * 2) / (values.length - 1) : 0;
  const pts = values.map((v, i) => {
    const x = pad + i * stepX;
    const y = pad + (1 - (v - min) / span) * (height - pad * 2);
    return [x, y] as const;
  });
  const line = pts
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`)
    .join(" ");
  const area = `${line} L${pts[pts.length - 1][0].toFixed(1)},${height} L${pts[0][0].toFixed(1)},${height} Z`;
  return { line, area };
}

export function Sparkline({
  values,
  positive,
  width = 72,
  height = 24,
}: {
  values: number[];
  positive: boolean | null;
  width?: number;
  height?: number;
}) {
  if (values.length < 2) return <span style={{ width, height }} className="block" />;
  const { line } = pathFor(values, width, height);
  const stroke =
    positive === null ? "var(--accent-text)" : positive ? "var(--gain)" : "var(--loss)";
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="block shrink-0"
      aria-hidden="true"
    >
      <path d={line} fill="none" stroke={stroke} strokeWidth="1.5" />
    </svg>
  );
}

export function PriceAreaChart({
  values,
  height = 120,
}: {
  values: number[];
  height?: number;
}) {
  const width = 560;
  if (values.length < 2) {
    return (
      <div
        className="flex items-center justify-center rounded-md border border-dashed border-border text-[12.5px] text-muted-foreground"
        style={{ height }}
      >
        No price history available
      </div>
    );
  }
  const { line, area } = pathFor(values, width, height, 4);
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="block w-full"
      style={{ height }}
      role="img"
      aria-label="Price history chart"
    >
      <defs>
        <linearGradient id="price-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#c9a96a" stopOpacity="0.28" />
          <stop offset="100%" stopColor="#c9a96a" stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <path d={area} fill="url(#price-fill)" />
      <path d={line} fill="none" stroke="#c9a96a" strokeWidth="1.8" />
    </svg>
  );
}
