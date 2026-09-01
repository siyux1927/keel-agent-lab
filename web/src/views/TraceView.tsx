import { useCallback, useEffect, useState } from 'react'
import { api, num } from '../api'
import { SpanTree } from '../components/SpanTree'
import { Card, Tag } from '../components/ui'
import type { SpanNode, TraceSummary } from '../types'

export function TraceView({ active }: { active: boolean }) {
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [tree, setTree] = useState<SpanNode[] | null>(null)

  const load = useCallback(async () => {
    const d = await api.traces(25)
    setTraces(d.traces)
  }, [])

  // 每次切到这个 tab 都重新拉一次, 否则跑完任务回来看到的还是上一次的列表
  useEffect(() => { if (active) void load() }, [active, load])

  const openTree = async (id: string) => {
    const d = await api.trace(id)
    setTree(d.tree)
  }

  return (
    <>
      <Card title="Trace 列表" actions={<button className="ghost" onClick={() => void load()}>刷新</button>}>
        <table style={{ marginTop: 10 }}>
          <thead>
            <tr>
              <th>Trace</th><th>名称</th><th>Spans</th><th>LLM</th><th>Tool</th>
              <th>Tokens</th><th>成本</th><th>错误</th>
            </tr>
          </thead>
          <tbody>
            {traces.length === 0 && <tr><td colSpan={8} className="empty">暂无数据</td></tr>}
            {traces.map(t => (
              <tr key={t.trace_id} style={{ cursor: 'pointer' }} onClick={() => void openTree(t.trace_id)}>
                <td className="small muted">{t.trace_id.slice(-8)}</td>
                <td>{t.name}</td>
                <td>{t.totals.spans}</td>
                <td>{t.totals.llm_calls}</td>
                <td>{t.totals.tool_calls}</td>
                <td>{num(t.totals.total_tokens)}</td>
                <td>${(t.totals.cost_usd ?? 0).toFixed(5)}</td>
                <td>
                  {t.totals.errors
                    ? <Tag kind="err">{t.totals.errors}</Tag>
                    : <Tag kind="ok">0</Tag>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Span 树" style={{ marginTop: 16 }}>
        <SpanTree tree={tree} />
      </Card>
    </>
  )
}
