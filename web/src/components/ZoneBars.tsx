import { num } from '../api'

export function ZoneBars({ zones }: { zones: Record<string, number> | null }) {
  if (!zones) return <div className="small muted">执行后显示各分区 token 分配</div>

  const entries = Object.entries(zones).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1])
  if (!entries.length) return <div className="small muted">执行后显示各分区 token 分配</div>

  const total = entries.reduce((sum, [, v]) => sum + v, 0) || 1

  return (
    <div className="small">
      {entries.map(([name, value]) => (
        <div key={name} style={{ marginBottom: 6 }}>
          <div className="row small">
            <span style={{ flex: 1 }}>{name}</span>
            <span className="muted">{num(value)}</span>
          </div>
          <div className="bar"><i style={{ width: `${(value / total * 100).toFixed(1)}%` }} /></div>
        </div>
      ))}
    </div>
  )
}
