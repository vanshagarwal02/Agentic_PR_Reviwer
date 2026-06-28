/**
 * api.js — Fetch wrapper for the AI Code Reviewer backend.
 *
 * All requests go to /api/reviews/* which Vite proxies to http://localhost:8000.
 * Every function returns the parsed JSON on success, or throws an Error.
 */

const BASE = '/api'

async function apiFetch(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`API ${res.status}: ${detail}`)
  }
  return res.json()
}

/**
 * Fetch aggregate stats for the header stat cards.
 * @param {string|null} repo  e.g. "owner/repo", or null for all
 */
export function fetchStats(repo = null) {
  const qs = repo ? `?repo=${encodeURIComponent(repo)}` : ''
  return apiFetch(`/reviews/stats${qs}`)
}

/**
 * Fetch the daily risk trend for the line chart.
 * @param {string|null} repo
 * @param {number} days  Number of days to look back (default 30)
 */
export function fetchTrend(repo = null, days = 30) {
  const params = new URLSearchParams({ days })
  if (repo) params.set('repo', repo)
  return apiFetch(`/reviews/trend?${params}`)
}

/**
 * Fetch a paginated list of reviews for the table.
 * @param {object} filters  { repo, verdict, risk_level, limit, offset }
 */
export function fetchReviews({ repo, verdict, risk_level, limit = 20, offset = 0 } = {}) {
  const params = new URLSearchParams({ limit, offset })
  if (repo)       params.set('repo', repo)
  if (verdict)    params.set('verdict', verdict)
  if (risk_level) params.set('risk_level', risk_level)
  return apiFetch(`/reviews?${params}`)
}

/**
 * Fetch the full detail for a single review (includes raw_json / inline comments).
 * @param {number} id
 */
export function fetchReviewById(id) {
  return apiFetch(`/reviews/${id}`)
}
