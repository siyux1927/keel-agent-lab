import type { MetricGoal } from '../types'

const W = 240, H = 42, PAD = 4

// 内联 SVG 画折线, 不引第三方图表库 —— 这个项目的依赖清单是刻意压到最小的。
export function Sparkline({ values, goal }: { values: number[]; goal: MetricGoal }) {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = (max - min) || 1
  const pts = values.map((v, i) => {
    const x = PAD + i * (W - PAD * 2) / Math.max(1, values.length - 1)
    const y = H - PAD - (v - min) / span * (H - PAD * 2)
    return [x, y] as const
  })

  const line = pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ')

  return (
    <svg width={W} height={H} style={{ display: 'block' }}>
      <polyline points={line} fill="none" stroke="#3a4a66" strokeWidth={1.5} />
      {pts.map(([x, y], i) => {
        let color = '#8b9ab3'
        if (i > 0 && values[i] !== values[i - 1]) {
          const better = goal === 'lower' ? values[i] < values[i - 1] : values[i] > values[i - 1]
          color = better ? '#3ecf8e' : '#ff6b6b'
        }
        return <circle key={i} cx={x.toFixed(1)} cy={y.toFixed(1)} r={2.5} fill={color} />
      })}
    </svg>
  )
}
