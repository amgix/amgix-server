import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { defineConfig } from 'vite'

const __dirname = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  // Relative URLs in built HTML (`./assets/...`) so the same dist works at any mount path; no subpath hardcoded.
  base: './',
  resolve: {
    alias: {
      '@amgix/amgix-client': resolve(__dirname, 'client/src/index.ts'),
    },
  },
  server: {
    proxy: {
      '/v1': {
        target: 'http://localhost:8234',
        changeOrigin: true,
      },
    },
  },
  preview: {
    proxy: {
      '/v1': {
        target: 'http://localhost:8234',
        changeOrigin: true,
      },
    },
  },
})
