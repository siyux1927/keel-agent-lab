import type {
  BenchHistory, ChunkLabResult, ContextLabResult, Health,
  IngestResult, MemoryDump, SearchHit, SpanNode, TraceSummary,
} from './types'

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(await r.text())
  return r.json() as Promise<T>
}

async function post<T>(url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json() as Promise<T>
}

export const api = {
  health: () => get<Health>('/api/health'),

  traces: (limit = 25) => get<{ traces: TraceSummary[] }>(`/api/traces?limit=${limit}`),
  trace: (id: string) => get<{ tree: SpanNode[] }>(`/api/trace/${encodeURIComponent(id)}`),

  memory: (sessionId: string, limit = 100) =>
    get<MemoryDump>(`/api/memory?session_id=${encodeURIComponent(sessionId)}&limit=${limit}`),
  memorySearch: (query: string, topK = 6) =>
    get<{ results: SearchHit[] }>(`/api/memory/search?query=${encodeURIComponent(query)}&top_k=${topK}`),
  memoryIngest: (text: string, source: string, strategy: string) =>
    post<IngestResult>('/api/memory/ingest', { text, source, strategy }),
  memoryDecay: () => post<{ archived: number }>('/api/memory/decay'),
  memoryReflect: (sessionId: string) =>
    post<{ insights?: string[] }>(`/api/memory/reflect?session_id=${encodeURIComponent(sessionId)}`),

  labContext: (task: string, sessionId: string, contextWindow: number) =>
    post<ContextLabResult>('/api/lab/context', {
      task, session_id: sessionId, context_window: contextWindow,
    }),
  labChunk: (text: string, chunkTokens: number, overlap: number) =>
    post<ChunkLabResult>('/api/lab/chunk', { text, chunk_tokens: chunkTokens, overlap }),

  benchHistory: (limit = 40) => get<BenchHistory>(`/api/bench/history?limit=${limit}`),
}

export function chatStreamUrl(message: string, sessionId: string, mode: string): string {
  return `/api/chat/stream?message=${encodeURIComponent(message)}`
    + `&session_id=${encodeURIComponent(sessionId)}&mode=${encodeURIComponent(mode)}`
}

export function benchStreamUrl(group: string): string {
  return `/api/bench/stream?group=${encodeURIComponent(group)}`
}

export const num = (n: number | undefined | null): string => (n ?? 0).toLocaleString()
