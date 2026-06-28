/**
 * ReviewTable.jsx
 *
 * Sortable, paginated, filterable table of agent reviews.
 *
 * Props:
 *   reviews       — array of review rows
 *   total         — total count (for pagination)
 *   loading       — bool
 *   page          — current page (0-indexed)
 *   pageSize      — rows per page
 *   onPageChange  — (newPage) => void
 *   onRowClick    — (reviewId) => void
 *   filterVerdict   — currently selected verdict filter
 *   filterRisk      — currently selected risk filter
 *   onFilterVerdict — (val) => void
 *   onFilterRisk    — (val) => void
 */

import { useState } from 'react'

const VERDICT_BADGE = {
  REQUEST_CHANGES: 'badge badge-request',
  APPROVE:         'badge badge-approve',
  COMMENT:         'badge badge-comment',
}
const VERDICT_LABEL = {
  REQUEST_CHANGES: '⛔ Request Changes',
  APPROVE:         '✅ Approve',
  COMMENT:         '💬 Comment',
}
const RISK_BADGE = {
  HIGH:   'badge badge-high',
  MEDIUM: 'badge badge-medium',
  LOW:    'badge badge-low',
}

function formatDate(ts) {
  if (!ts) return '—'
  const d = new Date(ts)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) +
    ' ' + d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
}

function ConfCell({ confidence }) {
  const pct = Math.round((confidence || 0) * 100)
  return (
    <div className="conf-cell">
      <div className="conf-bar-bg">
        <div className="conf-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>{pct}%</span>
    </div>
  )
}

const COLUMNS = [
  { key: 'created_at', label: 'Date' },
  { key: 'pr_number',  label: 'PR #' },
  { key: 'pr_title',   label: 'Title' },
  { key: 'repo',       label: 'Repo' },
  { key: 'verdict',    label: 'Verdict' },
  { key: 'risk_level', label: 'Risk' },
  { key: 'confidence', label: 'Confidence' },
  { key: 'inline_count', label: 'Comments' },
  { key: 'feedback',   label: '👍' },
]

export default function ReviewTable({
  reviews, total, loading, page, pageSize,
  onPageChange, onRowClick,
  filterVerdict, filterRisk, onFilterVerdict, onFilterRisk,
}) {
  const [sortKey, setSortKey] = useState('created_at')
  const [sortDir, setSortDir] = useState('desc')

  const totalPages = Math.ceil((total || 0) / pageSize)

  function handleSort(key) {
    if (key === sortKey) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  // Client-side sort the current page
  const sorted = [...(reviews || [])].sort((a, b) => {
    let av = a[sortKey], bv = b[sortKey]
    if (typeof av === 'string') av = av.toLowerCase()
    if (typeof bv === 'string') bv = bv.toLowerCase()
    if (av < bv) return sortDir === 'asc' ? -1 : 1
    if (av > bv) return sortDir === 'asc' ? 1 : -1
    return 0
  })

  const sortIcon = (key) => {
    if (key !== sortKey) return <span style={{ opacity: 0.3 }}>⇅</span>
    return sortDir === 'asc' ? '↑' : '↓'
  }

  return (
    <div className="glass-card table-card">
      {/* Toolbar */}
      <div className="table-toolbar">
        <span className="chart-title">Review History</span>
        <div className="filter-bar">
          <span className="filter-label">Filter:</span>
          <select
            id="filter-verdict"
            className="filter-select"
            value={filterVerdict}
            onChange={e => { onFilterVerdict(e.target.value); onPageChange(0) }}
          >
            <option value="">All Verdicts</option>
            <option value="REQUEST_CHANGES">Request Changes</option>
            <option value="APPROVE">Approve</option>
            <option value="COMMENT">Comment</option>
          </select>
          <select
            id="filter-risk"
            className="filter-select"
            value={filterRisk}
            onChange={e => { onFilterRisk(e.target.value); onPageChange(0) }}
          >
            <option value="">All Risk Levels</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {total} review{total !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              {COLUMNS.map(col => (
                <th key={col.key} onClick={() => handleSort(col.key)}>
                  {col.label} {sortIcon(col.key)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={COLUMNS.length}>
                  <div className="spinner-wrap"><div className="spinner" /></div>
                </td>
              </tr>
            ) : sorted.length === 0 ? (
              <tr>
                <td colSpan={COLUMNS.length}>
                  <div className="empty-state">
                    <span className="empty-icon">🔍</span>
                    <span>No reviews found — try clearing the filters</span>
                  </div>
                </td>
              </tr>
            ) : sorted.map(r => (
              <tr key={r.id} onClick={() => onRowClick(r.id)} title="Click for full detail">
                <td>{formatDate(r.created_at)}</td>
                <td className="primary">
                  <span className="mono" style={{ color: 'var(--cyan-400)' }}>
                    #{r.pr_number}
                  </span>
                </td>
                <td className="primary" style={{ maxWidth: 200 }}>
                  <span style={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {r.pr_title || '(no title)'}
                  </span>
                </td>
                <td>
                  <span className="mono" style={{ fontSize: '0.75rem' }}>
                    {r.repo || '—'}
                  </span>
                </td>
                <td>
                  <span className={VERDICT_BADGE[r.verdict] || 'badge badge-comment'}>
                    {VERDICT_LABEL[r.verdict] || r.verdict}
                  </span>
                </td>
                <td>
                  <span className={RISK_BADGE[r.risk_level] || 'badge'}>
                    {r.risk_level}
                  </span>
                </td>
                <td><ConfCell confidence={r.confidence} /></td>
                <td style={{ textAlign: 'center' }}>
                  <span style={{
                    background: 'rgba(139,92,246,0.15)',
                    color: 'var(--purple-400)',
                    padding: '0.15rem 0.5rem',
                    borderRadius: '99px',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                  }}>
                    {r.inline_count}
                  </span>
                </td>
                <td style={{ textAlign: 'center', fontSize: '1rem' }}>
                  {r.feedback === 1 ? '👍' : r.feedback === -1 ? '👎' : <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>—</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="pagination">
          <button
            id="page-prev"
            className="page-btn"
            disabled={page === 0}
            onClick={() => onPageChange(page - 1)}
          >
            ← Prev
          </button>

          {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
            const pageNum = totalPages <= 7 ? i
              : page < 4 ? i
              : page > totalPages - 5 ? totalPages - 7 + i
              : page - 3 + i
            return (
              <button
                key={pageNum}
                id={`page-${pageNum}`}
                className={`page-btn${pageNum === page ? ' active' : ''}`}
                onClick={() => onPageChange(pageNum)}
              >
                {pageNum + 1}
              </button>
            )
          })}

          <button
            id="page-next"
            className="page-btn"
            disabled={page >= totalPages - 1}
            onClick={() => onPageChange(page + 1)}
          >
            Next →
          </button>

          <span className="page-info">
            {page * pageSize + 1}–{Math.min((page + 1) * pageSize, total)} of {total}
          </span>
        </div>
      )}
    </div>
  )
}
