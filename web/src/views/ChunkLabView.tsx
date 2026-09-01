import { useState } from 'react'
import { api } from '../api'
import { SAMPLE_DOC } from '../sample'
import { Card, Tag } from '../components/ui'
import type { ChunkLabResult } from '../types'

export function ChunkLabView() {
  const [text, setText] = useState('')
  const [size, setSize] = useState('220')
  const [overlap, setOverlap] = useState('40')
  const [result, setResult] = useState<ChunkLabResult | null>(null)
  const [busy, setBusy] = useState(false)

  const run = async () => {
    const body = text.trim()
    if (!body) return
    setBusy(true)
    try {
      setResult(await api.labChunk(body, parseInt(size) || 220, parseInt(overlap) || 40))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <Card title="五种切片策略并排对比">
        <textarea rows={8} value={text} onChange={e => setText(e.target.value)} placeholder="粘贴一段文档..." />
        <div className="row" style={{ marginTop: 10 }}>
          <label className="small muted">块大小</label>
          <input type="number" value={size} onChange={e => setSize(e.target.value)} style={{ width: 100 }} />
          <label className="small muted">重叠</label>
          <input type="number" value={overlap} onChange={e => setOverlap(e.target.value)} style={{ width: 100 }} />
          <button className="act" onClick={() => void run()} disabled={busy}>对比</button>
          <button className="ghost" onClick={() => setText(SAMPLE_DOC)}>填充示例文本</button>
        </div>
      </Card>

      <div style={{ marginTop: 16 }}>
        {result && Object.entries(result.strategies).map(([name, r]) => {
          const s = r.stats
          return (
            <Card key={name} style={{ marginBottom: 14 }}>
              <div className="row">
                <b style={{ fontSize: 15 }}>{name}</b>
                <Tag>{s.count} 块</Tag>
                <Tag>平均 {s.avg_tokens} tok</Tag>
                <Tag>范围 {s.min_tokens}–{s.max_tokens}</Tag>
                <Tag kind={s.std_tokens > s.avg_tokens * 0.5 ? 'warn' : 'ok'}>标准差 {s.std_tokens}</Tag>
                {s.parents ? <Tag>父块 {s.parents} / 子块 {s.children}</Tag> : null}
              </div>
              <table style={{ marginTop: 8 }}>
                <thead><tr><th>#</th><th>tokens</th><th>标题路径</th><th>预览</th></tr></thead>
                <tbody>
                  {r.chunks.map(c => (
                    <tr key={c.index}>
                      <td>{c.index}</td>
                      <td>{c.tokens}</td>
                      <td className="small muted">
                        {c.title}{c.parent_id && <> <Tag>child</Tag></>}
                      </td>
                      <td className="small">{c.preview}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )
        })}
      </div>
    </>
  )
}
