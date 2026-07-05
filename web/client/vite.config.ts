import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import type { IncomingMessage, ServerResponse } from "node:http";

// wllama needs SharedArrayBuffer, which requires cross-origin isolation.
function setIsolation(res: ServerResponse) {
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
}
const crossOriginIsolation = {
  name: "cross-origin-isolation",
  configureServer(server: { middlewares: { use: (fn: unknown) => void } }) {
    server.middlewares.use((_req: IncomingMessage, res: ServerResponse, next: () => void) => {
      setIsolation(res);
      next();
    });
  },
  configurePreviewServer(server: { middlewares: { use: (fn: unknown) => void } }) {
    server.middlewares.use((_req: IncomingMessage, res: ServerResponse, next: () => void) => {
      setIsolation(res);
      next();
    });
  },
};

export default defineConfig({
  build: { outDir: "dist", target: "esnext", chunkSizeWarningLimit: 5000, emptyOutDir: true },
  server: {
    port: 5173,
    // Dev: forward API + SSE to the Deno engine server.
    proxy: { "/api": { target: "http://localhost:8787", changeOrigin: true } },
    // The wizard imports the dependency-free engine from ../../src.
    fs: { allow: ["../..", "."] },
  },
  optimizeDeps: { exclude: ["@wllama/wllama"] },
  worker: { format: "es" },
  plugins: [react(), crossOriginIsolation],
});
