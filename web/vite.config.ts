import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 产物直接写进 keel/server/static 并提交进版本库: 让 clone 下来 pip install 就能跑,
// 看项目的人不必先装一套 Node 工具链。代价是产物可能落后于源码, 由 CI 的陈旧检查兜底。
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../keel/server/static',
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
