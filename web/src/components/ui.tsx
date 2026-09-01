import { Fragment } from 'react'
import type { CSSProperties, ReactNode } from 'react'

export function Card(
  { title, children, style, actions }:
  { title?: ReactNode; children: ReactNode; style?: CSSProperties; actions?: ReactNode },
) {
  return (
    <div className="card" style={style}>
      {title !== undefined && (
        actions
          ? <div className="row"><h3 style={{ margin: 0, flex: 1 }}>{title}</h3>{actions}</div>
          : <h3>{title}</h3>
      )}
      {children}
    </div>
  )
}

export function Tag({ kind, children }: { kind?: 'ok' | 'err' | 'warn'; children: ReactNode }) {
  return <span className={kind ? `tag ${kind}` : 'tag'}>{children}</span>
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>
}

// .kv 是两列 grid, 两个格子必须是网格的直接子元素, 所以这里只能用 Fragment 而非包一层 div
export function KeyValueGrid({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <div className="kv">
      {rows.map(([k, v]) => (
        <Fragment key={k}>
          <b>{k}</b>
          <span>{v}</span>
        </Fragment>
      ))}
    </div>
  )
}
