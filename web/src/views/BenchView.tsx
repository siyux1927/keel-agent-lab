import { useCallback, useEffect, useRef, useState } from 'react'
import { api, benchStreamUrl } from '../api'
import { Sparkline } from '../components/Sparkline'
import { Card, Empty, Tag } from '../components/ui'
import type { BenchEvent, BenchRun, MetricGoal } from '../types'

const GOAL_LABEL: Record<MetricGoal, string> = {
  lower: '越小越好', higher: '越大越好', info: '仅参考',
}

const GROUPS: [string, string][] = [
  ['all', '全部六组（约 30 秒）'],
  ['guard', '护栏 A/B'],
  ['breaker', '工具熔断器'],
  ['mmr', 'MMR 去冗余'],
  ['budget', '预算调度 + 压缩'],
  ['digest', '动作清单注入'],
  ['dag', 'DAG 并发编排'],
]

function LatestMetrics({ runs }: { runs: BenchRun[] }) {
  const latest = runs[0]
  if (!latest) return <Empty>尚无实验记录</Empty>
  const prev = runs[1]
  const m = latest.meta ?? {}

  return (
    <>
      <div className="small muted" style={{ marginBottom: 8 }}>
        commit <b>{m.git_commit ?? '?'}</b> · {m.generated_at ?? ''} ·{' '}
        {m.provider ?? ''}/{m.model ?? ''} · 耗时 {m.duration_s ?? '?'}s{' '}
        {m.git_dirty && <Tag kind="warn">工作区未提交</Tag>}
      </div>
      <table>
        <thead><tr><th>指标</th><th>值</th><th>方向</th><th>较上次</th></tr></thead>
        <tbody>
          {Object.entries(latest.metrics).map(([name, v]) => {
            const before = prev?.metrics[name]
            let delta = <span className="muted">—</span>
            if (before && v.goal !== 'info' && before.value !== v.value) {
              const better = v.goal === 'lower' ? v.value < before.value : v.value > before.value
              const pct = before.value === 0
                ? (v.value - before.value).toFixed(2)
                : `${((v.value - before.value) / Math.abs(before.value) * 100).toFixed(1)}%`
              delta = <Tag kind={better ? 'ok' : 'err'}>{pct}</Tag>
            }
            return (
              <tr key={name}>
                <td className="small">{name}</td>
                <td>{v.value}{v.unit ?? ''}</td>
                <td className="small muted">{GOAL_LABEL[v.goal] ?? v.goal}</td>
                <td>{delta}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </>
  )
}

function Trend({ runs }: { runs: BenchRun[] }) {
  // 接口给的是倒序(最新在前), 时间正序才是"演进"
  const ordered = runs.slice().reverse()
  if (ordered.length < 2) return <Empty>至少需要两次实验记录才能画趋势</Empty>

  const names = [...new Set(ordered.flatMap(r => Object.keys(r.metrics)))]
    .filter(n => ordered.some(r => r.metrics[n] && r.metrics[n].goal !== 'info'))

  const cards = names.map(name => {
    // 只取该指标存在的那些次, 缺失的跳过而不是补 0 —— 补 0 会画出根本没发生过的悬崖
    const series = ordered.filter(r => r.metrics[name])
    if (series.length < 2) return null
    const values = series.map(r => r.metrics[name].value)
    const goal = series[series.length - 1].metrics[name].goal
    const first = values[0]
    const last = values[values.length - 1]
    const better = goal === 'lower' ? last < first : last > first

    return (
      <div className="card" key={name} style={{ padding: 12 }}>
        <div className="small" style={{ marginBottom: 2 }}>{name}</div>
        <div className="small muted" style={{ marginBottom: 6 }}>
          {GOAL_LABEL[goal]} · {series.length} 次记录 ·{' '}
          {last !== first
            ? <Tag kind={better ? 'ok' : 'err'}>{first} → {last}</Tag>
            : <Tag>无变化</Tag>}
        </div>
        <Sparkline values={values} goal={goal} />
      </div>
    )
  }).filter(Boolean)

  if (!cards.length) return <Empty>没有可画趋势的指标</Empty>

  return (
    <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill,minmax(268px,1fr))' }}>
      {cards}
    </div>
  )
}

export function BenchView({ active }: { active: boolean }) {
  const [runs, setRuns] = useState<BenchRun[]>([])
  const [group, setGroup] = useState('all')
  const [canRun, setCanRun] = useState(false)
  const [status, setStatus] = useState('')
  const [log, setLog] = useState('')
  const [running, setRunning] = useState(false)

  const sourceRef = useRef<EventSource | null>(null)
  const logRef = useRef<HTMLPreElement>(null)

  const load = useCallback(async () => {
    const d = await api.benchHistory(40)
    setRuns(d.runs ?? [])
    setCanRun(d.can_run)
    setStatus(d.can_run
      ? `已留档 ${(d.runs ?? []).length} 次`
      : '公开演示站不开放实验触发，以下为历史留档')
  }, [])

  useEffect(() => { if (active) void load() }, [active, load])
  useEffect(() => () => sourceRef.current?.close(), [])

  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [log])

  const run = () => {
    sourceRef.current?.close()
    setRunning(true)
    setLog('')
    setStatus('实验进行中…')

    const source = new EventSource(benchStreamUrl(group))
    sourceRef.current = source
    source.onmessage = ev => {
      let d: BenchEvent
      try { d = JSON.parse(ev.data) as BenchEvent } catch { return }
      if (d.type === 'log') {
        setLog(prev => prev + d.line + '\n')
      } else if (d.type === 'error') {
        setStatus('失败：' + d.error)
        source.close()
        setRunning(false)
      } else if (d.type === 'done') {
        setStatus('完成，已留档 ' + d.file)
        source.close()
        setRunning(false)
        void load()
      }
    }
    source.onerror = () => {
      source.close()
      setRunning(false)
      setStatus(prev => (prev === '实验进行中…' ? '连接中断' : prev))
    }
  }

  return (
    <>
      <Card title="A/B 消融实验">
        <div className="small muted" style={{ marginBottom: 12 }}>
          每个机制都做关/开对照，用数据回答「这套护栏到底有没有用」。
          指标自带方向（越小越好 / 越大越好 / 仅参考），历次结果留档，可以看出某次改动让哪个指标变差了。
        </div>
        <div className="row">
          <label className="small muted">实验组</label>
          <select value={group} onChange={e => setGroup(e.target.value)} style={{ width: 220 }}>
            {GROUPS.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
          </select>
          <button className="act" onClick={run} disabled={running || !canRun}>开始实验</button>
          <button className="ghost" onClick={() => void load()}>刷新历史</button>
          <span className="small muted">{status}</span>
        </div>
      </Card>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <Card title="最近一次的指标"><LatestMetrics runs={runs} /></Card>
        <Card title="执行日志">
          <pre ref={logRef} style={{ maxHeight: 520 }}>
            {log || '点击「开始实验」后在此实时输出'}
          </pre>
        </Card>
      </div>

      <Card title="指标趋势（按 commit 演进）" style={{ marginTop: 16 }}>
        <div className="small muted" style={{ marginBottom: 10 }}>
          只画有方向的指标。绿点表示比上一次更好，红点表示更差。
          墙钟耗时这类仅参考指标不画——它受机器负载影响，画出来只会制造噪声。
        </div>
        <Trend runs={runs} />
      </Card>
    </>
  )
}
