/**
 * App.jsx — AI Code Reviewer Dashboard
 *
 * Layout:
 *   [Header] — logo, live badge, repo selector
 *   [Stats Row] — 4 animated stat cards
 *   [Charts Row] — Risk Trend SVG chart | Verdict/Risk breakdown bars
 *   [Review Table] — sortable, paginated, filterable history
 *   [ReviewDetail drawer] — slides in on row click
 *
 * Data fetching:
 *   - fetchStats()  → every 30s (auto-refresh)
 *   - fetchTrend()  → on repo change
 *   - fetchReviews() → on page/filter change
 *   - fetchReviewById() → on row click (inside ReviewDetail)
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { fetchStats, fetchTrend, fetchReviews } from './api'
import RiskTrendChart  from './components/RiskTrendChart'
import RepoStatsCard   from './components/RepoStatsCard'
import ReviewTable     from './components/ReviewTable'
import ReviewDetail    from './components/ReviewDetail'

const PAGE_SIZE = 15
const REFRESH_INTERVAL = 30_000  // 30 seconds

// ── Animated counter hook ───────────────────────────────────────────────────
function useAnimatedCount(target) {
  const [val, setVal] = useState(0)
  const prev = useRef(0)
  useEffect(() => {
    if (target === prev.current) return
    const start = prev.current
    const end = target || 0
    const diff = end - start
    if (diff === 0) return
    const steps = 20
    const step = diff / steps
    let current = start
    let count = 0
    const timer = setInterval(() => {
      count++
      current += step
      setVal(Math.round(current))
      if (count >= steps) {
        setVal(end)
        clearInterval(timer)
        prev.current = end
      }
    }, 40)
    return () => clearInterval(timer)
  }, [target])
  return val
}

// ── Stat Card ───────────────────────────────────────────────────────────────
function StatCard({ icon, value, label, sub, animatedValue }) {
  const displayed = animatedValue !== undefined ? animatedValue : value
  return (
    <div className="stat-card">
      <span className="stat-icon">{icon}</span>
      <span className="stat-value">{displayed}</span>
      <span className="stat-label">{label}</span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  )
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  // ── Data state
  const [stats, setStats]    = useState(null)
  const [trend, setTrend]    = useState([])
  const [reviews, setReviews] = useState([])
  const [total, setTotal]    = useState(0)

  // ── Loading / error state
  const [statsLoading, setStatsLoading]   = useState(true)
  const [trendLoading, setTrendLoading]   = useState(true)
  const [tableLoading, setTableLoading]   = useState(true)
  const [statsError, setStatsError]       = useState(null)

  // ── UI state
  const [selectedRepo, setSelectedRepo]   = useState('')
  const [filterVerdict, setFilterVerdict] = useState('')
  const [filterRisk, setFilterRisk]       = useState('')
  const [page, setPage]                   = useState(0)
  const [activeReviewId, setActiveReviewId] = useState(null)

  // ── Animated stats
  const animTotal    = useAnimatedCount(stats?.total)
  const animApprove  = useAnimatedCount(stats?.by_verdict?.APPROVE ?? 0)
  const animFlagged  = useAnimatedCount(stats?.by_verdict?.REQUEST_CHANGES ?? 0)
  const animAvgConf  = useAnimatedCount(Math.round((stats?.avg_confidence ?? 0) * 100))

  // ── Load stats (+ auto-refresh) ──────────────────────────────────────────
  const loadStats = useCallback(() => {
    setStatsError(null)
    fetchStats(selectedRepo || null)
      .then(data => { setStats(data); setStatsLoading(false) })
      .catch(e => { setStatsError(e.message); setStatsLoading(false) })
  }, [selectedRepo])

  useEffect(() => {
    setStatsLoading(true)
    loadStats()
    const interval = setInterval(loadStats, REFRESH_INTERVAL)
    return () => clearInterval(interval)
  }, [loadStats])

  // ── Load trend ────────────────────────────────────────────────────────────
  useEffect(() => {
    setTrendLoading(true)
    fetchTrend(selectedRepo || null, 30)
      .then(data => { setTrend(data); setTrendLoading(false) })
      .catch(() => setTrendLoading(false))
  }, [selectedRepo])

  // ── Load reviews (table) ─────────────────────────────────────────────────
  useEffect(() => {
    setTableLoading(true)
    fetchReviews({
      repo: selectedRepo || undefined,
      verdict: filterVerdict || undefined,
      risk_level: filterRisk || undefined,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    })
      .then(data => {
        setReviews(data.reviews || [])
        setTotal(data.total || 0)
        setTableLoading(false)
      })
      .catch(() => setTableLoading(false))
  }, [selectedRepo, filterVerdict, filterRisk, page])

  // ── Repo list from stats ──────────────────────────────────────────────────
  const repos = stats?.repos ?? []

  // ── Handlers ─────────────────────────────────────────────────────────────
  function handleRepoChange(e) {
    setSelectedRepo(e.target.value)
    setPage(0)
    setFilterVerdict('')
    setFilterRisk('')
  }

  return (
    <div className="app">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon">🛡</div>
            <div>
              <div className="logo-text">AI Code Reviewer</div>
              <div className="logo-sub">Analytics Dashboard</div>
            </div>
          </div>
          <div className="header-right">
            <div className="live-badge">
              <div className="live-dot" />
              Live · {REFRESH_INTERVAL / 1000}s refresh
            </div>
          </div>
        </div>
      </header>

      {/* ── Main Content ───────────────────────────────────────────────────── */}
      <main className="main">

        {/* Connection error */}
        {statsError && (
          <div className="error-message">
            ⚠ Cannot reach backend: {statsError}
            <span style={{ marginLeft: 'auto', fontSize: '0.75rem', opacity: 0.7 }}>
              Is the server running on port 8000?
            </span>
          </div>
        )}

        {/* Repo filter */}
        <div className="filter-bar">
          <span className="filter-label">Repository:</span>
          <select
            id="repo-filter"
            className="filter-select"
            value={selectedRepo}
            onChange={handleRepoChange}
          >
            <option value="">All Repositories</option>
            {repos.map(r => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
          {selectedRepo && (
            <button
              className="page-btn"
              onClick={() => { setSelectedRepo(''); setPage(0) }}
              style={{ fontSize: '0.75rem' }}
            >
              ✕ Clear
            </button>
          )}
        </div>

        {/* ── Stats Row ──────────────────────────────────────────────────── */}
        <section>
          <p className="section-title">Overview</p>
          <div className="stats-row">
            <StatCard
              icon="📋"
              label="Total Reviews"
              animatedValue={animTotal}
              sub={selectedRepo ? `in ${selectedRepo}` : 'all repos'}
            />
            <StatCard
              icon="✅"
              label="PRs Approved"
              animatedValue={animApprove}
              sub={`${stats?.total ? Math.round((animApprove / stats.total) * 100) : 0}% of total`}
            />
            <StatCard
              icon="⛔"
              label="Changes Requested"
              animatedValue={animFlagged}
              sub={`${stats?.total ? Math.round((animFlagged / stats.total) * 100) : 0}% of total`}
            />
            <StatCard
              icon="🎯"
              label="Avg Confidence"
              animatedValue={`${animAvgConf}%`}
              sub="across all reviews"
            />
          </div>
        </section>

        {/* ── Charts Row ─────────────────────────────────────────────────── */}
        <section>
          <p className="section-title">Trends & Distribution</p>
          <div className="charts-row">
            <RiskTrendChart data={trend} loading={trendLoading} />
            <RepoStatsCard  stats={stats} loading={statsLoading} />
          </div>
        </section>

        {/* ── Review Table ───────────────────────────────────────────────── */}
        <section>
          <ReviewTable
            reviews={reviews}
            total={total}
            loading={tableLoading}
            page={page}
            pageSize={PAGE_SIZE}
            onPageChange={setPage}
            onRowClick={id => setActiveReviewId(id)}
            filterVerdict={filterVerdict}
            filterRisk={filterRisk}
            onFilterVerdict={v => { setFilterVerdict(v); setPage(0) }}
            onFilterRisk={v => { setFilterRisk(v); setPage(0) }}
          />
        </section>

      </main>

      {/* ── Review Detail Drawer ───────────────────────────────────────────── */}
      <ReviewDetail
        reviewId={activeReviewId}
        onClose={() => setActiveReviewId(null)}
      />
    </div>
  )
}
