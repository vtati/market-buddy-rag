// All API calls are same-origin: next.config.ts rewrites /api/* to the
// API server (API_PROXY_TARGET, default http://localhost:8000). This
// keeps the browser off cross-origin/CORS and off any third-party
// hostname — the deployed site only ever talks to its own domain.
export const API_BASE = "";

export interface Citation {
  doc: string;
  section: string;
  excerpt: string;
  span: [number, number];
  score: number;
}

export interface TraceStep {
  node: string;
  latency_ms: number;
  detail: string;
}

/** What one chat turn resolved to — the client sends it back as the
 *  next question's context so follow-ups keep their company. */
export interface ResolvedContext {
  ticker: string | null;
  fiscal_year: number | null;
}

export interface ChatResponse {
  answer: string;
  refused: boolean;
  clarification: boolean;
  citations: Citation[];
  trace: TraceStep[];
  mode: "llm" | "extractive";
  latency_ms: number;
  strategy: string;
  retrieval_mode: string;
  corpus: string;
  resolved: ResolvedContext | null;
  /** The standalone rewrite when a follow-up was contextualized. */
  understood_as: string | null;
}

export type DocSource = "bundled" | "precached" | "live";

export interface CorpusDoc {
  doc_id: string;
  ticker: string;
  label: string;
  chars: number;
  sections: string[];
  source: DocSource;
}

export interface SessionCompany {
  ticker: string;
  name: string;
  cik: string;
  source: DocSource;
  accession: string;
  fiscal_year: number;
  filed: string;
  report_date: string;
  doc_label: string;
  items: string[];
  chars: number;
  chunks: { fixed: number; section: number };
  figures_rows: number;
}

export interface CorpusResponse {
  documents: CorpusDoc[];
  total_chars: number;
  chunks: Record<string, number>;
  generation_mode: "llm" | "extractive";
  session_companies: SessionCompany[];
}

export interface IngestCandidate {
  ticker: string;
  name: string;
  cik: string;
}

export interface IngestStage {
  name: string;
  status: "pending" | "active" | "done" | "error";
  detail: string;
}

export interface IngestStatus {
  job_id: string;
  ticker: string;
  stage: string;
  stages: IngestStage[];
  result: SessionCompany | null;
  error: string | null;
}

export interface ComparisonRow {
  strategy: string;
  mode: string;
  chunks: number;
  hit_at_4: number;
  mrr: number;
  mean_latency_ms: number;
  per_question: { id: string; hit_rank: number | null; latency_ms: number }[];
}

export interface ComparisonResponse {
  generated_at: string;
  corpus_documents: string[];
  question_count: number;
  embedding_model: string;
  reranker_model: string;
  rows: ComparisonRow[];
}

export interface EvalRow {
  id: string;
  type: string;
  question: string;
  expected: string;
  refused: boolean;
  answer: string;
  citations: { doc: string; section: string }[];
  faithful: boolean;
  faithfulness_detail: string;
  latency_ms: number;
  pass: boolean;
  note: string;
}

export interface EvalResponse {
  generated_at: string;
  total: number;
  passed: number;
  pass_rate: number;
  faithfulness_rate_answered: number;
  mean_latency_ms: number;
  rows: EvalRow[];
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${path} failed: ${res.status}`);
  return (await res.json()) as T;
}

export interface MarketQuote {
  ticker: string;
  name: string;
  price: number;
  change: number | null;
  change_pct: number | null;
  prev_close: number | null;
  day_high: number | null;
  day_low: number | null;
  volume: number | null;
  market_cap: number | null;
  pe_ratio: number | null;
  high_52w: number | null;
  low_52w: number | null;
  currency: string;
  source: "yahoo" | "stooq" | null;
  as_of: string;
  delayed: boolean;
  available: boolean;
  spark?: number[];
}

export interface MarketOverview {
  indices: MarketQuote[];
  companies: MarketQuote[];
  source: "yahoo" | "stooq" | null;
  as_of: string;
  delayed: boolean;
  available: boolean;
}

export interface MarketHistory {
  ticker: string;
  range: string;
  points: { date: string; close: number }[];
  source: "yahoo" | "stooq" | null;
  as_of: string;
  available: boolean;
}

export interface TrendingItem {
  ticker: string;
  name: string;
  price: number | null;
  change_pct: number | null;
}

export interface MarketTrending {
  trending: TrendingItem[];
  source: string | null;
  as_of: string;
  available: boolean;
}

export interface FilingFigureRow {
  fiscal_year: number;
  revenue: number | null;
  net_income: number | null;
}

export interface FilingFigures {
  ticker: string;
  rows: FilingFigureRow[];
  available: boolean;
  source: string;
}

export const fetchMarketOverview = () =>
  getJSON<MarketOverview>("/api/market/overview");
export const fetchMarketQuote = (ticker: string) =>
  getJSON<MarketQuote>(`/api/market/quote?ticker=${encodeURIComponent(ticker)}`);
export const fetchMarketHistory = (ticker: string, range: string) =>
  getJSON<MarketHistory>(
    `/api/market/history?ticker=${encodeURIComponent(ticker)}&range=${range}`,
  );
export const fetchMarketTrending = () =>
  getJSON<MarketTrending>("/api/market/trending");
export const fetchFilingFigures = (ticker: string) =>
  getJSON<FilingFigures>(
    `/api/market/filing-figures?ticker=${encodeURIComponent(ticker)}`,
  );

export const fetchCorpus = () => getJSON<CorpusResponse>("/api/corpus");
export const fetchComparison = () =>
  getJSON<ComparisonResponse>("/api/results/comparison");
export const fetchEval = () => getJSON<EvalResponse>("/api/results/eval");

async function postJSON<T>(path: string, payload: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function postChat(
  question: string,
  strategy: string,
  mode: string,
  corpus: string = "active",
  context?: ResolvedContext | null,
): Promise<ChatResponse> {
  return postJSON<ChatResponse>("/api/chat", {
    question,
    strategy,
    mode,
    corpus,
    ...(context?.ticker ? { context } : {}),
  });
}

export const resolveCompany = (query: string) =>
  postJSON<{ candidates: IngestCandidate[] }>("/api/ingest/resolve", { query });

export const startIngest = (ticker: string) =>
  postJSON<{ job_id: string; ticker: string }>("/api/ingest", { ticker });

export const fetchIngestStatus = (jobId: string) =>
  getJSON<IngestStatus>(`/api/ingest/status/${jobId}`);
