import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In Docker, VITE_API_BASE is set to http://api:8000 (service name).
// Locally it falls back to http://localhost:8000.
const apiTarget = process.env.VITE_API_BASE || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
