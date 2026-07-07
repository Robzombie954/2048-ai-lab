/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // host: true binds 0.0.0.0 so phones/tablets on the same LAN can reach it
  // at http://<your-LAN-IP>:5173. IP-based hosts are allowed by default.
  server: { host: true, port: 5173, strictPort: true },
  test: {
    environment: 'node',
    include: ['tests/**/*.spec.ts'],
  },
})
