import { num } from '../api'
import type { SpanNode } from '../types'
import { Empty } from './ui'

function Node({ node, depth }: { node: SpanNode; depth: number }) {
  const kind = node.status === 'error' ? 'err' : node.kind === 'llm' ? 'warn' : 'ok'
  const tokens = (node.prompt_tokens + node.completion_tokens) || 0
  return (
    <>
      <div className="node" style={{ paddingLeft: depth * 22 }}>
        <span className={`tag ${kind}`}>{node.kind}</span> {node.name}{' '}
        <span className="muted">
          {node.duration_ms}ms
          {tokens ? ` · ${num(tokens)}tok` : ''}
          {node.error ? ` · ${node.error}` : ''}
        </span>
      </div>
      {(node.children ?? []).map((child, i) => (
        <Node key={`${child.name}-${i}`} node={child} depth={depth + 1} />
      ))}
    </>
  )
}

export function SpanTree({ tree }: { tree: SpanNode[] | null }) {
  if (!tree) return <Empty>点击上方任一 trace 查看</Empty>
  if (!tree.length) return <Empty>该 trace 没有 span</Empty>
  return (
    <div className="tree">
      {tree.map((node, i) => <Node key={`${node.name}-${i}`} node={node} depth={0} />)}
    </div>
  )
}
