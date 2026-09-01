export type ViewId = 'chat' | 'trace' | 'memory' | 'ctxlab' | 'chunklab' | 'bench'

export type MetricGoal = 'lower' | 'higher' | 'info'

export interface Health {
  provider: string
  model: string
  offline_mode: boolean
  memory: { active: number; total: number }
}

export interface Usage {
  cost_usd?: number
  llm_calls?: number
  tool_calls?: number
  prompt_tokens?: number
  completion_tokens?: number
  wall_ms?: number
  errors?: number
}

// ---------------------------------------------------------------------------
// 对话 SSE 事件
//
// 这个联合类型就是前后端之间的契约。服务端 emit 一个新事件却忘了在这里登记,
// switch 的 default 分支会立刻暴露它, 而不是在页面上静默丢失。
// ---------------------------------------------------------------------------

interface FinishFields {
  answer?: string
  usage?: Usage
  status?: string
  mode?: string
  stop_reason?: string
  speedup?: number
}

export interface PlanNode {
  id: string
  title: string
  depends_on: string[]
}

export type ChatEvent =
  | { type: 'agent.start'; session_id: string; goal: string }
  | { type: 'orchestrator.start'; goal: string }
  | { type: 'skill.recalled'; pattern: string; win_rate?: number; steps?: string[] }
  | { type: 'plan.ready'; nodes?: PlanNode[] }
  | { type: 'dag.start'; nodes: number; levels: number; plan: unknown }
  | { type: 'dag.node'; id: string; status: string; title?: string; duration_ms?: number; error?: string }
  | {
      type: 'dag.finish'
      done: number; failed: number; skipped: number
      speedup: number; serial_ms: number; wall_ms: number
    }
  | { type: 'context.built'; index: number; tokens: number; zones?: Record<string, number> }
  | { type: 'thought'; text: string }
  | { type: 'tool.start'; tool: string; args: unknown }
  | { type: 'tool.end'; tool: string; ok: boolean; latency_ms: number; preview: string }
  | { type: 'tool.blocked'; tool: string }
  | { type: 'guard'; action: string; reason: string }
  | { type: 'reflect'; replans: number; diagnosis: string; new_plan?: string[] }
  | { type: 'verify.rejected'; issues?: string[] }
  | {
      type: 'critic.verdict'
      round: number; accepted: boolean; score: number
      issues?: string[]; suggestion?: string
    }
  | { type: 'memory.reflected'; insights?: string[] }
  | { type: 'answer.degraded'; reason: string }
  | ({ type: 'agent.finish' } & FinishFields)
  | ({ type: 'orchestrator.finish' } & FinishFields)
  | { type: 'error'; error: string }
  | { type: 'done' }

export type ChatFinishEvent = Extract<ChatEvent, { type: 'agent.finish' | 'orchestrator.finish' }>

// ---------------------------------------------------------------------------
// Trace
// ---------------------------------------------------------------------------

export interface TraceTotals {
  spans: number
  llm_calls: number
  tool_calls: number
  total_tokens: number
  cost_usd: number
  errors: number
}

export interface TraceSummary {
  trace_id: string
  name: string
  totals: TraceTotals
}

export interface SpanNode {
  kind: string
  name: string
  status: string
  duration_ms: number
  prompt_tokens: number
  completion_tokens: number
  error?: string
  children?: SpanNode[]
}

// ---------------------------------------------------------------------------
// 记忆
// ---------------------------------------------------------------------------

export type MemoryLayer = 'working' | 'episodic' | 'semantic' | 'procedural'

export interface MemoryItem {
  content?: string
  importance?: number
  strength?: number
  access_count?: number
  source?: string
  archived?: boolean
}

export interface LayerStats {
  active?: number
  archived?: number
  avg_strength?: number
}

export interface MemoryDump {
  by_layer: Record<string, MemoryItem[]>
  stats: { by_layer: Record<string, LayerStats> }
}

export interface SearchHit {
  layer: string
  content: string
  score: number
  breakdown: { vector?: number; bm25?: number; recency?: number; importance?: number }
}

export interface ChunkStats {
  count: number
  avg_tokens: number
  min_tokens: number
  max_tokens: number
  std_tokens: number
  parents?: number
  children?: number
}

export interface IngestResult {
  indexed: number
  strategy: string
  chunk_stats: ChunkStats
}

// ---------------------------------------------------------------------------
// 上下文实验室
// ---------------------------------------------------------------------------

export interface BudgetZone {
  name: string
  requested: number
  granted: number
  dropped: boolean
  needs_compression: boolean
  reason?: string
}

export interface ContextLabResult {
  budget: {
    total_budget: number
    granted_total: number
    utilization: number
    requested_total: number
    pressure: number
    rounds: number
    zones: BudgetZone[]
  }
  zones: Record<string, { method?: string }>
  retrieved_items: { zone: string; source: string; score: number; tokens: number; text: string }[]
  messages: { role: string; content: string; meta: { zone?: string } }[]
}

// ---------------------------------------------------------------------------
// 切片实验室
// ---------------------------------------------------------------------------

export interface ChunkPreview {
  index: number
  tokens: number
  title: string
  parent_id?: string
  preview: string
}

export interface ChunkLabResult {
  strategies: Record<string, { stats: ChunkStats; chunks: ChunkPreview[] }>
}

// ---------------------------------------------------------------------------
// 消融实验
// ---------------------------------------------------------------------------

export interface BenchMetric {
  value: number
  unit?: string
  goal: MetricGoal
}

export interface BenchRun {
  meta: {
    git_commit?: string
    generated_at?: string
    provider?: string
    model?: string
    duration_s?: number
    git_dirty?: boolean
  }
  metrics: Record<string, BenchMetric>
}

export interface BenchHistory {
  runs: BenchRun[]
  can_run: boolean
}

export type BenchEvent =
  | { type: 'log'; line: string }
  | { type: 'error'; error: string }
  | { type: 'done'; file: string }
