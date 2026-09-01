import { useCallback, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { api } from './api'
import { BenchView } from './views/BenchView'
import { ChatView } from './views/ChatView'
import { ChunkLabView } from './views/ChunkLabView'
import { ContextLabView } from './views/ContextLabView'
import { MemoryView } from './views/MemoryView'
import { TraceView } from './views/TraceView'
import type { Health, ViewId } from './types'

const TABS: [ViewId, string][] = [
  ['chat', '对话 / 执行'],
  ['trace', '调用链追踪'],
  ['memory', '记忆浏览器'],
  ['ctxlab', '上下文实验室'],
  ['chunklab', '切片实验室'],
  ['bench', '消融实验'],
]

function View({ id, active, children }: { id: ViewId; active: ViewId; children: ReactNode }) {
  return <section className={id === active ? 'view active' : 'view'}>{children}</section>
}

export function App() {
  const [view, setView] = useState<ViewId>('chat')
  // session id 被对话、记忆浏览器、上下文实验室三处共用, 所以提到应用层
  const [sessionId, setSessionId] = useState('default')
  const [health, setHealth] = useState<Health | null>(null)
  const [cost, setCost] = useState(0)

  const loadHealth = useCallback(async () => {
    try { setHealth(await api.health()) } catch { setHealth(null) }
  }, [])

  useEffect(() => {
    void loadHealth()
    const timer = setInterval(() => void loadHealth(), 15000)
    return () => clearInterval(timer)
  }, [loadHealth])

  return (
    <>
      <header>
        <div className="logo">Ke<span>el</span></div>
        <span className={health && !health.offline_mode ? 'badge on' : 'badge'}>
          {health
            ? `${health.provider} · ${health.model} (${health.offline_mode ? '离线' : '在线'})`
            : '服务未就绪'}
        </span>
        <span className="badge">
          {health ? `记忆 ${health.memory.active}/${health.memory.total}` : 'memory …'}
        </span>
        <span className="badge">cost ${cost.toFixed(4)}</span>
        <div style={{ flex: 1 }} />
        <span className="badge small">上下文工程 · 记忆 · Loop Engineering · 编排</span>
      </header>

      <nav>
        {TABS.map(([id, label]) => (
          <button
            key={id}
            className={view === id ? 'active' : undefined}
            onClick={() => setView(id)}
          >
            {label}
          </button>
        ))}
      </nav>

      <main>
        <View id="chat" active={view}>
          <ChatView
            sessionId={sessionId}
            onSessionId={setSessionId}
            onCost={setCost}
            onFinished={() => void loadHealth()}
          />
        </View>
        <View id="trace" active={view}><TraceView active={view === 'trace'} /></View>
        <View id="memory" active={view}>
          <MemoryView
            sessionId={sessionId}
            active={view === 'memory'}
            onChanged={() => void loadHealth()}
          />
        </View>
        <View id="ctxlab" active={view}><ContextLabView sessionId={sessionId} /></View>
        <View id="chunklab" active={view}><ChunkLabView /></View>
        <View id="bench" active={view}><BenchView active={view === 'bench'} /></View>
      </main>
    </>
  )
}
