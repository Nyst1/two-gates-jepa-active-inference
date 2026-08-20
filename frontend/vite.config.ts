import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const standalone = mode === "standalone";
  return {
    base: standalone ? "./" : "/",
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": "http://127.0.0.1:8000",
      },
    },
    build: standalone
      ? {
          outDir: "standalone-dist",
          emptyOutDir: true,
          cssCodeSplit: false,
          assetsInlineLimit: Number.MAX_SAFE_INTEGER,
          rolldownOptions: { output: { codeSplitting: false } },
        }
      : undefined,
  };
});
