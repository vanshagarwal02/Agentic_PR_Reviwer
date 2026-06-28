/**
 * RiskTrendChart.jsx
 *
 * SVG line chart showing daily LOW / MEDIUM / HIGH review counts.
 * Built from scratch — no charting library, pure SVG + React.
 *
 * Props:
 *   data  — array of { date, LOW, MEDIUM, HIGH }
 *   loading — bool
 */

import { useMemo } from 'react'

const COLORS = {
  HIGH:   '#f87171',
  MEDIUM: '#fb923c',
  LOW:    '#4ade80',
}

const PAD = { top: 16, right: 16, bottom: 32, left: 36 }
const W = 600  // viewBox width
const H = 160  // viewBox height

function linePoints(data, key, maxVal) {
  const innerW = W - PAD.left - PAD.right
  const innerH = H - PAD.top - PAD.bottom
  if (!data.length) return ''

  return data.map((d, i) => {
    const x = PAD.left + (i / (data.length - 1 || 1)) * innerW
    const y = PAD.top + innerH - (maxVal > 0 ? (d[key] / maxVal) * innerH : 0)
    return `${x},${y}`
  }).join(' ')
}

export default function RiskTrendChart({ data = [], loading }) {
  const maxVal = useMemo(() => {
    if (!data.length) return 5
    return Math.max(5, ...data.flatMap(d => [d.HIGH, d.MEDIUM, d.LOW]))
  }, [data])

  // X-axis: show every 7th date label
  const innerW = W - PAD.left - PAD.right
  const innerH = H - PAD.top - PAD.bottom

  const xLabels = useMemo(() => {
    if (!data.length) return []
    const step = Math.ceil(data.length / 6)
    return data
      .map((d, i) => ({ d, i }))
      .filter(({ i }) => i % step === 0 || i === data.length - 1)
  }, [data])

  // Y gridlines
  const yTicks = [0, Math.round(maxVal / 2), maxVal]

  return (
    <div className="glass-card chart-card">
      <div className="chart-header">
        <span className="chart-title">Risk Trend (30 days)</span>
        <div className="chart-legend">
          {['HIGH', 'MEDIUM', 'LOW'].map(k => (
            <span key={k} className="legend-item">
              <span className="legend-dot" style={{ background: COLORS[k] }} />
              {k}
            </span>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="spinner-wrap"><div className="spinner" /></div>
      ) : !data.length ? (
        <div className="empty-state">
          <span className="empty-icon">📈</span>
          <span>No trend data yet — run some reviews first</span>
        </div>
      ) : (
        <svg
          className="trend-svg"
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
        >
          {/* Grid lines */}
          {yTicks.map(tick => {
            const y = PAD.top + innerH - (maxVal > 0 ? (tick / maxVal) * innerH : 0)
            return (
              <g key={tick}>
                <line
                  x1={PAD.left} y1={y} x2={W - PAD.right} y2={y}
                  stroke="rgba(255,255,255,0.06)" strokeWidth="1"
                />
                <text x={PAD.left - 6} y={y + 4} fill="rgba(148,163,184,0.6)"
                  fontSize="10" textAnchor="end">
                  {tick}
                </text>
              </g>
            )
          })}

          {/* X-axis labels */}
          {xLabels.map(({ d, i }) => {
            const x = PAD.left + (i / (data.length - 1 || 1)) * innerW
            const label = d.date.slice(5) // MM-DD
            return (
              <text key={d.date} x={x} y={H - 4}
                fill="rgba(148,163,184,0.6)" fontSize="10" textAnchor="middle">
                {label}
              </text>
            )
          })}

          {/* Area fills (subtle) */}
          {['LOW', 'MEDIUM', 'HIGH'].map(key => {
            const points = linePoints(data, key, maxVal)
            if (!points) return null
            const firstX = PAD.left
            const lastX  = PAD.left + innerW
            const baseY  = PAD.top + innerH
            return (
              <polygon
                key={`area-${key}`}
                points={`${firstX},${baseY} ${points} ${lastX},${baseY}`}
                fill={COLORS[key]}
                fillOpacity="0.07"
              />
            )
          })}

          {/* Lines */}
          {['LOW', 'MEDIUM', 'HIGH'].map(key => {
            const points = linePoints(data, key, maxVal)
            if (!points) return null
            return (
              <polyline
                key={`line-${key}`}
                points={points}
                fill="none"
                stroke={COLORS[key]}
                strokeWidth="2"
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity="0.9"
              />
            )
          })}

          {/* Dots at last data point */}
          {['LOW', 'MEDIUM', 'HIGH'].map(key => {
            if (!data.length) return null
            const last = data[data.length - 1]
            const x = PAD.left + innerW
            const y = PAD.top + innerH - (maxVal > 0 ? (last[key] / maxVal) * innerH : 0)
            if (last[key] === 0) return null
            return (
              <circle key={`dot-${key}`} cx={x} cy={y} r="3"
                fill={COLORS[key]} />
            )
          })}
        </svg>
      )}
    </div>
  )
}
