/**
 * ReviewDetail.jsx
 *
 * Slide-in drawer showing full detail of a single review.
 * Includes meta-grid, summary, inline comments, and tools used.
 *
 * Props:
 *   reviewId  — number | null (null = drawer closed)
 *   onClose   — function
 */

import { useEffect, useState, useCallback } from 'react'
import { fetchReviewById } from '../api'

async function postFeedback(reviewId, score) {
  const res = await fetch(`/api/reviews/${reviewId}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ score }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

const SEVERITY_EMOJI = { CRITICAL: '🔴', HIGH: '🟠', MEDIUM: '🟡', LOW: '🔵' }
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
const VERDICT_BADGE = {
  REQUEST_CHANGES: 'badge badge-request',
  APPROVE:         'badge badge-approve',
  COMMENT:         'badge badge-comment',
}

export default function ReviewDetail({ reviewId, onClose }) {
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [feedback, setFeedback] = useState(0)   // 1=👍 -1=👎 0=none
  const [fbSaved, setFbSaved] = useState(false)

  useEffect(() => {
    if (!reviewId) return
    setLoading(true)
    setError(null)
    setDetail(null)

    fetchReviewById(reviewId)
      .then(data => {
        setDetail(data)
        setFeedback(data.feedback ?? 0)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [reviewId])

  // Close on Escape
  useEffect(() => {
    const handler = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  if (!reviewId) return null

  const rawJson = detail?.raw_json || {}
  const inlineComments = rawJson.inline_comments || []
  const toolsUsed = rawJson.tools_used || detail?.tools_used || []

  return (
    <div className="modal-overlay" onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal-drawer" role="dialog" aria-modal="true">
        {/* Header */}
        <div className="modal-header">
          <div>
            <div className="modal-title">
              Review #{detail?.pr_number ?? reviewId}
              {detail?.pr_title ? ` — ${detail.pr_title}` : ''}
            </div>
            <div className="modal-sub">{detail?.pr_url || '...'}</div>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">✕</button>
        </div>

        {/* Body */}
        <div className="modal-body">
          {loading && <div className="spinner-wrap"><div className="spinner" /></div>}
          {error && <div className="error-message">⚠ {error}</div>}

          {detail && (
            <>
              {/* Meta grid */}
              <div>
                <p className="detail-section-title">Overview</p>
                <div className="detail-meta-grid">
                  <div className="meta-item">
                    <div className="meta-key">Verdict</div>
                    <div className="meta-val">
                      <span className={VERDICT_BADGE[detail.verdict] || 'badge badge-comment'}>
                        {VERDICT_LABEL[detail.verdict] || detail.verdict}
                      </span>
                    </div>
                  </div>
                  <div className="meta-item">
                    <div className="meta-key">Risk Level</div>
                    <div className="meta-val">
                      <span className={RISK_BADGE[detail.risk_level] || 'badge'}>
                        {detail.risk_level}
                      </span>
                    </div>
                  </div>
                  <div className="meta-item">
                    <div className="meta-key">Confidence</div>
                    <div className="meta-val">{Math.round((detail.confidence || 0) * 100)}%</div>
                  </div>
                  <div className="meta-item">
                    <div className="meta-key">PR Type</div>
                    <div className="meta-val">{detail.pr_type || '—'}</div>
                  </div>
                  <div className="meta-item">
                    <div className="meta-key">Inline Comments</div>
                    <div className="meta-val">{detail.inline_count ?? inlineComments.length}</div>
                  </div>
                  <div className="meta-item">
                    <div className="meta-key">Elapsed</div>
                    <div className="meta-val">{detail.elapsed_sec ? `${detail.elapsed_sec.toFixed(1)}s` : '—'}</div>
                  </div>
                  <div className="meta-item" style={{ gridColumn: '1 / -1' }}>
                    <div className="meta-key">Repository</div>
                    <div className="meta-val">{detail.repo || '—'}</div>
                  </div>
                  <div className="meta-item" style={{ gridColumn: '1 / -1' }}>
                    <div className="meta-key">Reviewed At</div>
                    <div className="meta-val">
                      {detail.created_at
                        ? new Date(detail.created_at).toLocaleString()
                        : '—'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Tools Used */}
              {toolsUsed.length > 0 && (
                <div>
                  <p className="detail-section-title">Tools Used</p>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                    {toolsUsed.map(t => (
                      <span key={t} className="badge badge-comment mono"
                        style={{ fontSize: '0.7rem' }}>
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Summary */}
              {detail.summary && (
                <div>
                  <p className="detail-section-title">Agent Summary</p>
                  <p className="summary-text">{detail.summary}</p>
                </div>
              )}

              {/* Inline Comments */}
              {inlineComments.length > 0 && (
                <div>
                  <p className="detail-section-title">
                    Inline Comments ({inlineComments.length})
                  </p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {inlineComments.map((c, i) => (
                      <div key={i} className="inline-comment">
                        <div className="inline-comment-header">
                          <span className="badge"
                            style={{
                              background: severityBg(c.severity),
                              color: severityColor(c.severity),
                              border: `1px solid ${severityColor(c.severity)}40`,
                              fontSize: '0.68rem',
                            }}>
                            {SEVERITY_EMOJI[c.severity] || '🔵'} {c.severity}
                          </span>
                          <span className="badge badge-comment" style={{ fontSize: '0.68rem' }}>
                            {c.category}
                          </span>
                          <span className="inline-file">{c.file}</span>
                          <span className="inline-line">:{c.line}</span>
                        </div>
                        <p className="inline-comment-body">{c.comment}</p>
                        {c.suggestion && (
                          <pre className="inline-suggestion">{c.suggestion}</pre>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 👍 👎 Feedback */}
              <div style={{
                borderTop: '1px solid var(--border)',
                paddingTop: '1.25rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
              }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Was this review helpful?
                </span>
                {[{ score: 1, emoji: '👍', label: 'Yes' }, { score: -1, emoji: '👎', label: 'No' }].map(({ score, emoji, label }) => (
                  <button
                    key={score}
                    id={`feedback-${score === 1 ? 'up' : 'down'}-${reviewId}`}
                    onClick={async () => {
                      const next = feedback === score ? 0 : score
                      try {
                        await postFeedback(reviewId, next)
                        setFeedback(next)
                        setFbSaved(true)
                        setTimeout(() => setFbSaved(false), 2000)
                      } catch {}
                    }}
                    style={{
                      background: feedback === score
                        ? score === 1 ? 'rgba(74,222,128,0.15)' : 'rgba(248,113,113,0.15)'
                        : 'var(--bg-glass)',
                      border: `1px solid ${
                        feedback === score
                          ? score === 1 ? 'rgba(74,222,128,0.4)' : 'rgba(248,113,113,0.4)'
                          : 'var(--border)'
                      }`,
                      borderRadius: 'var(--radius-sm)',
                      color: feedback === score
                        ? score === 1 ? 'var(--risk-low)' : 'var(--risk-high)'
                        : 'var(--text-secondary)',
                      padding: '0.35rem 0.75rem',
                      fontSize: '0.85rem',
                      cursor: 'pointer',
                      transition: 'var(--transition)',
                      fontFamily: 'inherit',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem',
                    }}
                  >
                    {emoji} {label}
                  </button>
                ))}
                {fbSaved && (
                  <span style={{
                    fontSize: '0.72rem',
                    color: 'var(--risk-low)',
                    animation: 'fade-in 0.2s ease',
                  }}>
                    ✓ Saved
                  </span>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function severityColor(s) {
  return { CRITICAL: '#f87171', HIGH: '#fb923c', MEDIUM: '#fbbf24', LOW: '#60a5fa' }[s] || '#94a3b8'
}
function severityBg(s) {
  return { CRITICAL: 'rgba(248,113,113,0.12)', HIGH: 'rgba(251,146,60,0.12)',
           MEDIUM: 'rgba(251,191,36,0.1)', LOW: 'rgba(96,165,250,0.1)' }[s] || 'rgba(148,163,184,0.1)'
}
