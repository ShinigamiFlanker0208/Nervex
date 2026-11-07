import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API binds to loopback and allows CORS for :5173, but proxying keeps the browser same-origin
// (no preflight, and uploads/polling behave exactly like production behind one reverse proxy).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/_health": { target: "http://127.0.0.1:8000", rewrite: () => "/" },
    },
  },
});
