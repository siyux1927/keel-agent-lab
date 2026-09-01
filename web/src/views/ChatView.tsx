import { useCallback, useEffect, useRef, useState } from 'react'
import { chatStreamUrl, num } from '../api'
import { EventStream, describe } from '../components/EventStream'
import type { StreamCard } from '../components/EventStream'
import { ZoneBars } from '../components/ZoneBars'
import { Card, KeyValueGrid } from '../components/ui'
import type { ChatEvent, ChatFinishEvent } from '../types'

const PRESETS: [string, string][] = [
  ['死循环检测', '死循环演示 请反复确认当前时间'],
  ['熔断降级', '熔断演示 调用不稳定服务取数据'],
  ['多工具协作', '帮我算一下 (128*7+56)/4, 再查一下知识库的发布窗口'],
]

function metricRows(e: ChatFinishEvent): [string, string][] {
  const u = e.usage ?? {}
  const rows: [string, string][] = [
    ['状态', e.status || e.mode || '完成'],
    ['终止原因', e.stop_reason || '目标达成'],
    ['LLM 调用', num(u.llm_calls)],
    ['工具调用', num(u.tool_calls)],
    ['Prompt tokens', num(u.prompt_tokens)],
    ['Completion tokens', num(u.completion_tokens)],
    ['成本(USD)', (u.cost_usd ?? 0).toFixed(6)],
    ['耗时', `${Math.round(u.wall_ms ?? 0)} ms`],
    ['错误 span', num(u.errors)],
  ]
  if (e.speedup) rows.push(['并发加速比', `${e.speedup}×`])
  return rows
}

export function ChatView(
  { sessionId, onSessionId, onCost, onFinished }:
  { sessionId: string; onSessionId: (v: string) => void; onCost: (v: number) => void; onFinished: () => void },
) {
  const [message, setMessage] = useState('')
  const [mode, setMode] = useState('react')
  const [cards, setCards] = useState<StreamCard[]>([])
  const [zones, setZones] = useState<Record<string, number> | null>(null)
  const [answer, setAnswer] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<[string, string][]>([])
  const [running, setRunning] = useState(false)

  const sourceRef = useRef<EventSource | null>(null)
  const seqRef = useRef(0)

  // 组件卸载时必须断开, 否则切走再切回会留下一条永远在收数据的孤儿连接
  useEffect(() => () => sourceRef.current?.close(), [])

  const push = useCallback((card: Omit<StreamCard, 'id'>) => {
    setCards(prev => [...prev, { ...card, id: seqRef.current++ }])
  }, [])

  const handle = useCallback((e: ChatEvent) => {
    const card = describe(e)
    if (card) push(card)

    if (e.type === 'context.built' && e.zones) setZones(e.zones)
    if (e.type === 'agent.finish' || e.type === 'orchestrator.finish') {
      setAnswer(e.answer || '(无答案)')
      setMetrics(metricRows(e))
      onCost(e.usage?.cost_usd ?? 0)
    }
    if (e.type === 'error') setRunning(false)
    if (e.type === 'done') {
      setRunning(false)
      sourceRef.current?.close()
      onFinished()
    }
  }, [push, onCost, onFinished])

  const start = () => {
    const text = message.trim()
    if (!text) return
    sourceRef.current?.close()
    setCards([])
    setAnswer(null)
    setMetrics([])
    setRunning(true)

    const source = new EventSource(chatStreamUrl(text, sessionId, mode))
    sourceRef.current = source
    source.onmessage = ev => {
      try { handle(JSON.parse(ev.data) as ChatEvent) } catch { /* 半条 JSON, 丢掉即可 */ }
    }
    source.onerror = () => { source.close(); setRunning(false) }
  }

  const clear = () => {
    setCards([]); setAnswer(null); setMetrics([]); setZones(null)
  }

  return (
    <div className="grid g2">
      <Card title="任务执行">
        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder="例如: 查一下知识库里的发布窗口规定; 同时搜索 ReAct 论文的核心思想; 然后计算 (99+1)*3"
        />
        <div className="row" style={{ marginTop: 10 }}>
          <select value={mode} onChange={e => setMode(e.target.value)} style={{ width: 190 }}>
            <option value="react">ReAct 单智能体</option>
            <option value="orchestrate">DAG 多智能体编排</option>
          </select>
          <input
            value={sessionId}
            onChange={e => onSessionId(e.target.value)}
            style={{ width: 130 }}
            placeholder="session"
          />
          <button className="act" onClick={start} disabled={running}>执行</button>
          <button className="ghost" onClick={clear}>清空</button>
        </div>
        <div className="row small muted" style={{ marginTop: 10 }}>
          护栏演示:
          {PRESETS.map(([label, text]) => (
            <button key={label} className="ghost small" onClick={() => setMessage(text)}>{label}</button>
          ))}
        </div>
        <EventStream cards={cards} />
      </Card>

      <Card title="最终答案">
        <div className={answer ? 'answer' : 'answer muted'}>
          {answer ?? (running ? '执行中…' : '尚未执行')}
        </div>
        <h3 style={{ marginTop: 18 }}>运行指标</h3>
        {metrics.length
          ? <KeyValueGrid rows={metrics} />
          : <KeyValueGrid rows={[['状态', <span className="muted">—</span>]]} />}
        <h3 style={{ marginTop: 18 }}>上下文预算占用</h3>
        <ZoneBars zones={zones} />
      </Card>
    </div>
  )
}
