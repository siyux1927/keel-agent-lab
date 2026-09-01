import { useEffect, useRef } from 'react'
import type { ReactNode } from 'react'
import { num } from '../api'
import type { ChatEvent } from '../types'

export interface StreamCard {
  id: number
  cls: string
  title: string
  body?: ReactNode
}

/** 把一个 SSE 事件翻译成一张流式卡片; 返回 null 表示该事件只有副作用、不进事件流。 */
export function describe(e: ChatEvent): Omit<StreamCard, 'id'> | null {
  switch (e.type) {
    case 'agent.start':
      return { cls: '', title: `开始执行 · session ${e.session_id}`,
        body: <span className="muted small">{e.goal}</span> }
    case 'orchestrator.start':
      return { cls: '', title: '编排启动', body: <span className="muted small">{e.goal}</span> }
    case 'skill.recalled':
      return { cls: 'reflect', title: '召回程序性记忆', body: (
        <span className="small">
          匹配「{e.pattern}」胜率 {Math.round((e.win_rate ?? 0) * 100)}%
          {' · '}路径 {(e.steps ?? []).join(' → ')}
        </span>
      ) }
    case 'plan.ready':
      return { cls: '', title: `规划完成 · ${(e.nodes ?? []).length} 个子任务`, body: (
        <pre className="small">
          {(e.nodes ?? []).map(n => `${n.id} ${n.title}  deps=[${n.depends_on}]`).join('\n')}
        </pre>
      ) }
    case 'dag.start':
      return { cls: '', title: `DAG 调度 · ${e.nodes} 节点 / ${e.levels} 层`,
        body: <span className="small muted">并发层: {JSON.stringify(e.plan)}</span> }
    case 'dag.node':
      return { cls: 'tool', title: `节点 ${e.id} · ${e.status}`, body: (
        <span className="small muted">
          {e.title ?? ''}{e.duration_ms ? ` · ${e.duration_ms}ms` : ''}{e.error ? ` ${e.error}` : ''}
        </span>
      ) }
    case 'dag.finish':
      return { cls: 'done', title: 'DAG 完成', body: (
        <span className="small">
          完成 {e.done} · 失败 {e.failed} · 跳过 {e.skipped} · 并发加速 <b>{e.speedup}×</b>
          {' '}(串行 {e.serial_ms}ms → 实际 {e.wall_ms}ms)
        </span>
      ) }
    case 'context.built':
      return { cls: 'ctx', title: `第 ${e.index + 1} 步 · 上下文 ${num(e.tokens)} tokens`, body: (
        <span className="small muted">
          {Object.entries(e.zones ?? {}).filter(([, v]) => v > 0).map(([k, v]) => `${k}:${v}`).join('  ')}
        </span>
      ) }
    case 'thought':
      return { cls: 'thought', title: '思考', body: <span className="small">{e.text}</span> }
    case 'tool.start':
      return { cls: 'tool', title: `调用 ${e.tool}`,
        body: <span className="small muted">{JSON.stringify(e.args)}</span> }
    case 'tool.end':
      return { cls: 'tool', title: `${e.ok ? '✓ ' : '✗ '}${e.tool} · ${e.latency_ms}ms`,
        body: <span className="small muted">{e.preview}</span> }
    case 'tool.blocked':
      return { cls: 'guard', title: `动作被拦截 · ${e.tool}`,
        body: <span className="small">该动作已被反思判定无效</span> }
    case 'guard':
      return { cls: 'guard', title: `护栏触发 · ${e.action}`,
        body: <span className="small">{e.reason}</span> }
    case 'reflect':
      return { cls: 'reflect', title: `反思重规划 #${e.replans}`, body: (
        <>
          <div className="small">诊断: {e.diagnosis}</div>
          <div className="small muted">新计划: {(e.new_plan ?? []).join(' / ')}</div>
        </>
      ) }
    case 'verify.rejected':
      return { cls: 'guard', title: '自检未通过',
        body: <span className="small">{(e.issues ?? []).join('; ')}</span> }
    case 'critic.verdict':
      return { cls: e.accepted ? 'done' : 'guard',
        title: `质检第 ${e.round} 轮 · ${e.accepted ? '通过' : '打回'} (${e.score})`,
        body: <span className="small muted">{(e.issues ?? []).join('; ') || e.suggestion || ''}</span> }
    case 'memory.reflected':
      return { cls: 'reflect', title: '记忆固化',
        body: <span className="small">{(e.insights ?? []).join(' / ')}</span> }
    case 'answer.degraded':
      return { cls: 'guard', title: '降级作答', body: <span className="small">{e.reason}</span> }
    case 'error':
      return { cls: 'guard', title: '错误', body: <span className="small">{e.error}</span> }
    case 'agent.finish':
    case 'orchestrator.finish':
      return { cls: 'done', title: '执行结束' }
    case 'done':
      return null
  }
}

export function EventStream({ cards }: { cards: StreamCard[] }) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = ref.current
    if (el) el.scrollTop = el.scrollHeight
  }, [cards])

  return (
    <div className="stream" ref={ref} style={{ marginTop: 14 }}>
      {cards.map(c => (
        <div key={c.id} className={c.cls ? `ev ${c.cls}` : 'ev'}>
          <div className="t">{c.title}</div>
          {c.body && <div>{c.body}</div>}
        </div>
      ))}
    </div>
  )
}
