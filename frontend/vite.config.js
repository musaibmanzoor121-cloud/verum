import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite dev server + React plugin. Runs on http://localhost:5173 by default.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
})
