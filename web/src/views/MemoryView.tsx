import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import { Card, Empty, Tag } from '../components/ui'
import type { IngestResult, MemoryDump, SearchHit } from '../types'

const LAYER_NAMES: Record<string, string> = {
  working: '工作记忆(近期对话)',
  episodic: '情景记忆(事件流)',
  semantic: '语义记忆(事实/文档块)',
  procedural: '程序性记忆(技能)',
}

const STRATEGIES: [string, string][] = [
  ['structure', 'structure 结构感知'],
  ['recursive', 'recursive 递归'],
  ['semantic', 'semantic 语义'],
  ['hierarchical', 'hierarchical 分层'],
  ['fixed', 'fixed 定长'],
]

export function MemoryView(
  { sessionId, active, onChanged }:
  { sessionId: string; active: boolean; onChanged: () => void },
) {
  const [doc, setDoc] = useState('')
  const [strategy, setStrategy] = useState('structure')
  const [source, setSource] = useState('handbook.md')
  const [ingested, setIngested] = useState<IngestResult | null>(null)

  const [query, setQuery] = useState('')
  const [hits, setHits] = useState<SearchHit[] | null>(null)

  const [dump, setDump] = useState<MemoryDump | null>(null)
  const [notice, setNotice] = useState('')

  const load = useCallback(async () => {
    setDump(await api.memory(sessionId, 100))
  }, [sessionId])

  useEffect(() => { if (active) void load() }, [active, load])

  const ingest = async () => {
    const text = doc.trim()
    if (!text) return
    setIngested(await api.memoryIngest(text, source, strategy))
    await load()
    onChanged()
  }

  const search = async () => {
    const q = query.trim()
    if (!q) return
    const d = await api.memorySearch(q, 6)
    setHits(d.results)
  }

  const decay = async () => {
    const r = await api.memoryDecay()
    setNotice(`已归档 ${r.archived} 条低强度记忆`)
    await load()
  }

  const reflect = async () => {
    const r = await api.memoryReflect(sessionId)
    const insights = r.insights ?? []
    setNotice(insights.length ? `固化出 ${insights.length} 条洞察: ${insights.join(' / ')}` : '本次没有产出新洞察')
    await load()
  }

  return (
    <>
      <div className="grid g2">
        <Card title="知识入库(演示切片 → 向量化 → 语义记忆)">
          <textarea rows={8} value={doc} onChange={e => setDoc(e.target.value)} placeholder="粘贴 Markdown 文档..." />
          <div className="row" style={{ marginTop: 10 }}>
            <select value={strategy} onChange={e => setStrategy(e.target.value)} style={{ width: 170 }}>
              {STRATEGIES.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
            </select>
            <input value={source} onChange={e => setSource(e.target.value)} style={{ width: 150 }} />
            <button className="act" onClick={() => void ingest()}>入库</button>
          </div>
          {ingested && (
            <div className="small muted" style={{ marginTop: 10 }}>
              已索引 <b>{ingested.indexed}</b> 个块 · 策略 {ingested.strategy} · 平均{' '}
              {ingested.chunk_stats.avg_tokens} tokens (min {ingested.chunk_stats.min_tokens} / max{' '}
              {ingested.chunk_stats.max_tokens} / 标准差 {ingested.chunk_stats.std_tokens})
            </div>
          )}
        </Card>

        <Card title="混合检索(向量 + BM25 + 新鲜度 + 重要性)">
          <div className="row">
            <input value={query} onChange={e => setQuery(e.target.value)} placeholder="检索问题" style={{ flex: 1 }} />
            <button className="act" onClick={() => void search()}>检索</button>
          </div>
          <div style={{ marginTop: 12 }}>
            {hits === null ? null : hits.length === 0 ? <Empty>无结果</Empty> : (
              <table>
                <thead>
                  <tr>
                    <th>层</th><th>内容</th><th>总分</th><th>向量</th>
                    <th>BM25</th><th>新鲜度</th><th>重要性</th>
                  </tr>
                </thead>
                <tbody>
                  {hits.map((r, i) => (
                    <tr key={i}>
                      <td><Tag>{r.layer}</Tag></td>
                      <td className="small">{r.content.slice(0, 140)}</td>
                      <td><b>{r.score.toFixed(3)}</b></td>
                      <td>{(r.breakdown.vector ?? 0).toFixed(3)}</td>
                      <td>{(r.breakdown.bm25 ?? 0).toFixed(3)}</td>
                      <td>{(r.breakdown.recency ?? 0).toFixed(3)}</td>
                      <td>{(r.breakdown.importance ?? 0).toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Card>
      </div>

      <Card
        title="四层记忆"
        style={{ marginTop: 16 }}
        actions={
          <>
            <button className="ghost" onClick={() => void load()}>刷新</button>
            <button className="ghost" onClick={() => void decay()}>触发遗忘</button>
            <button className="ghost" onClick={() => void reflect()}>强制反思固化</button>
          </>
        }
      >
        {notice && <div className="small muted" style={{ marginBottom: 10 }}>{notice}</div>}
        {!dump ? <Empty>加载中</Empty> : Object.entries(dump.by_layer).map(([layer, items]) => {
          const st = dump.stats.by_layer[layer] ?? {}
          return (
            <div key={layer} style={{ marginBottom: 18 }}>
              <div className="row">
                <b>{LAYER_NAMES[layer] ?? layer}</b>
                <Tag>活跃 {st.active ?? items.length}</Tag>
                <Tag>归档 {st.archived ?? 0}</Tag>
                <Tag>平均强度 {st.avg_strength ?? '-'}</Tag>
              </div>
              {items.length === 0 ? (
                <div className="small muted" style={{ padding: '8px 0' }}>空</div>
              ) : (
                <table style={{ marginTop: 6 }}>
                  <thead>
                    <tr><th>内容</th><th>重要性</th><th>强度</th><th>命中</th><th>来源</th></tr>
                  </thead>
                  <tbody>
                    {items.slice(0, 12).map((m, i) => (
                      <tr key={i}>
                        <td className="small">
                          {(m.content ?? '').slice(0, 160)}
                          {m.archived && <> <Tag kind="err">已归档</Tag></>}
                        </td>
                        <td>{m.importance ?? '-'}</td>
                        <td>{m.strength ?? '-'}</td>
                        <td>{m.access_count ?? 0}</td>
                        <td className="small muted">{m.source ?? ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )
        })}
      </Card>
    </>
  )
}
