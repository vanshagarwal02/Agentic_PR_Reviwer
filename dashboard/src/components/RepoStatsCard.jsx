/**
 * RepoStatsCard.jsx
 *
 * Shows verdict breakdown (REQUEST_CHANGES / APPROVE / COMMENT) and risk
 * distribution as animated horizontal bars.
 *
 * Props:
 *   stats   — object from fetchStats()
 *   loading — bool
 */

const VERDICT_CONFIG = [
  { key: 'REQUEST_CHANGES', label: '⛔ Request Changes', color: '#f87171' },
  { key: 'APPROVE',         label: '✅ Approve',         color: '#4ade80' },
  { key: 'COMMENT',         label: '💬 Comment',         color: '#94a3b8' },
]

const RISK_CONFIG = [
  { key: 'HIGH',   label: '🔴 High',   color: '#f87171' },
  { key: 'MEDIUM', label: '🟠 Medium', color: '#fb923c' },
  { key: 'LOW',    label: '🟢 Low',    color: '#4ade80' },
]

function BarRow({ label, value, total, color }) {
  const pct = total > 0 ? Math.round((value / total) * 100) : 0
  return (
    <div className="verdict-row">
      <span className="verdict-label">{label}</span>
      <div className="verdict-bar-wrap">
        <div
          className="verdict-bar"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="verdict-count">{value ?? 0}</span>
    </div>
  )
}

export default function RepoStatsCard({ stats, loading }) {
  const total = stats?.total ?? 0
  const byVerdict = stats?.by_verdict ?? {}
  const byRisk = stats?.by_risk ?? {}

  return (
    <div className="glass-card chart-card" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="chart-header">
        <span className="chart-title">Review Breakdown</span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          {total} total
        </span>
      </div>

      {loading ? (
        <div className="spinner-wrap"><div className="spinner" /></div>
      ) : total === 0 ? (
        <div className="empty-state">
          <span className="empty-icon">📊</span>
          <span>No reviews yet</span>
        </div>
      ) : (
        <>
          <p className="detail-section-title">By Verdict</p>
          <div className="verdict-list" style={{ marginBottom: '1.25rem' }}>
            {VERDICT_CONFIG.map(({ key, label, color }) => (
              <BarRow
                key={key}
                label={label}
                value={byVerdict[key] ?? 0}
                total={total}
                color={color}
              />
            ))}
          </div>

          <p className="detail-section-title">By Risk Level</p>
          <div className="verdict-list">
            {RISK_CONFIG.map(({ key, label, color }) => (
              <BarRow
                key={key}
                label={label}
                value={byRisk[key] ?? 0}
                total={total}
                color={color}
              />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
