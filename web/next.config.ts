import type { NextConfig } from "next";

// The browser only ever calls this origin's /api/*; Next proxies those
// requests server-side to the FastAPI service (API_PROXY_TARGET). Local
// development defaults to the API on :8000.
const API_PROXY_TARGET = process.env.API_PROXY_TARGET ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API_PROXY_TARGET}/api/:path*` },
    ];
  },
};

export default nextConfig;
