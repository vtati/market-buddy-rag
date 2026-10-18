import type { Metadata } from "next";
import { Fraunces, Inter } from "next/font/google";
import { Nav } from "@/components/nav";
import "./globals.css";

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Market Buddy RAG — Financial Document Intelligence",
  description:
    "A RAG Q&A system over real SEC 10-K filings: two chunking strategies compared, hybrid retrieval with cross-encoder reranking, verified cited answers, and an honest refusal path. Gen Academy Week 2.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${fraunces.variable} ${inter.variable} dark`}>
      <body className="min-h-screen bg-background font-sans text-foreground">
        <Nav />
        <main className="mx-auto w-full max-w-7xl px-4 sm:px-6 lg:px-8">
          {children}
        </main>
        <footer className="mx-auto max-w-7xl px-4 pt-4 pb-10 sm:px-6 lg:px-8">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-border pt-5 text-[12px] text-muted-foreground">
            <span>
              Market Buddy RAG — a demo distilled from the Market Buddy research
              core. Gen Academy · Mastering Agentic AI · Week 2.
            </span>
            <span className="ml-auto">
              Corpus: SEC EDGAR 10-K excerpts — Apple, Microsoft, NVIDIA.
            </span>
          </div>
        </footer>
      </body>
    </html>
  );
}
