import { useState } from 'react'
import { api, num } from '../api'
import { Card, Empty, KeyValueGrid, Tag } from '../components/ui'
import type { ContextLabResult } from '../types'

export function ContextLabView({ sessionId }: { sessionId: string }) {
  const [task, setTask] = useState('总结知识库里关于发布流程和上下文预算的规定')
  const [window, setWindow] = useState('8192')
  const [result, setResult] = useState<ContextLabResult | null>(null)
  const [busy, setBusy] = useState(false)

  const run = async () => {
    setBusy(true)
    try {
      setResult(await api.labContext(task, sessionId, parseInt(window) || 8192))
    } finally {
      setBusy(false)
    }
  }

  const b = result?.budget

  return (
    <>
      <Card title="上下文组装可视化 —— 看清楚模型到底收到了什么">
        <div className="row">
          <input value={task} onChange={e => setTask(e.target.value)} placeholder="任务描述" style={{ flex: 1 }} />
          <input
            type="number" value={window} onChange={e => setWindow(e.target.value)}
            style={{ width: 120 }} title="上下文窗口"
          />
          <button className="act" onClick={() => void run()} disabled={busy}>组装</button>
        </div>
        <div className="small muted" style={{ marginTop: 8 }}>
          调小上下文窗口(比如 1500), 可以直接观察低优先级分区被丢弃、高优先级分区被压缩的过程。
        </div>
      </Card>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <Card title="预算分配">
          {!b || !result ? <Empty>待组装</Empty> : (
            <>
              <KeyValueGrid rows={[
                ['可用预算', `${num(b.total_budget)} tokens`],
                ['实际占用', `${num(b.granted_total)} (利用率 ${(b.utilization * 100).toFixed(1)}%)`],
                ['原始需求', `${num(b.requested_total)} (压力 ${(b.pressure * 100).toFixed(0)}%)`],
                ['注水轮次', String(b.rounds)],
              ]} />
              <table style={{ marginTop: 10 }}>
                <thead><tr><th>分区</th><th>需求</th><th>分配</th><th>处理</th></tr></thead>
                <tbody>
                  {b.zones.map(z => (
                    <tr key={z.name}>
                      <td>{z.name}</td>
                      <td>{num(z.requested)}</td>
                      <td>{num(z.granted)}</td>
                      <td>
                        {z.dropped
                          ? <Tag kind="err">整块丢弃</Tag>
                          : z.needs_compression
                            ? <Tag kind="warn">{result.zones[z.name]?.method ?? '压缩'}</Tag>
                            : <Tag kind="ok">原样</Tag>}
                        {' '}<span className="small muted">{z.reason ?? ''}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </Card>

        <Card title="检索来源(provenance)">
          {!result ? <Empty>待组装</Empty> : result.retrieved_items.length === 0 ? (
            <Empty>当前会话没有可召回的记忆, 先去「记忆浏览器」入库一些文档</Empty>
          ) : (
            <table>
              <thead><tr><th>分区</th><th>来源</th><th>分数</th><th>tokens</th><th>内容</th></tr></thead>
              <tbody>
                {result.retrieved_items.map((i, k) => (
                  <tr key={k}>
                    <td><Tag>{i.zone}</Tag></td>
                    <td className="small">{i.source}</td>
                    <td>{i.score.toFixed(3)}</td>
                    <td>{i.tokens}</td>
                    <td className="small muted">{i.text.slice(0, 110)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
      </div>

      <Card title="最终送入模型的消息" style={{ marginTop: 16 }}>
        <pre>
          {result
            ? result.messages
                .map(m => `───── [${m.meta.zone || m.role}] role=${m.role} ─────\n${m.content}`)
                .join('\n\n')
            : '待组装'}
        </pre>
      </Card>
    </>
  )
}
