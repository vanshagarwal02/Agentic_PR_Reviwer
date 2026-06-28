"""
db/database.py — Persistence layer for the AI Code Reviewer (Week 7).

WHY THIS EXISTS:
  Weeks 1–6 saved every agent review only to logs/agent_pr_N.json.
  That works for a single session but gives you no ability to:
    - Query review history across PRs or repos
    - See risk trends over time
    - Build a dashboard that survives server restarts

  This module writes every completed agent review to a SQLite database.
  SQLite requires zero setup — it's a single file (reviews.db) in the
  project root. To switch to PostgreSQL, set DATABASE_URL in .env:
    DATABASE_URL=postgresql://user:pass@localhost/ai_reviewer

HOW IT WORKS:
  - Uses Python's built-in `sqlite3` for SQLite (no extra dependencies).
  - For PostgreSQL: if DATABASE_URL starts with "postgresql://", uses
    psycopg2 (already in requirements.txt).
  - A thread-local connection pool avoids locking issues under FastAPI's
    async workers.

SCHEMA (single table — keeps Week 7 simple and portable):
  reviews
    id           INTEGER PK AUTOINCREMENT
    created_at   TIMESTAMP   — when the review was saved
    pr_url       TEXT        — full GitHub PR URL
    pr_number    INTEGER     — numeric PR id
    pr_title     TEXT        — PR title from GitHub
    repo         TEXT        — "owner/repo" string
    verdict      TEXT        — REQUEST_CHANGES | APPROVE | COMMENT
    risk_level   TEXT        — LOW | MEDIUM | HIGH
    confidence   REAL        — 0.0–1.0
    pr_type      TEXT        — feature | bugfix | refactor | ...
    tools_used   TEXT        — JSON array ["scan_for_vulnerabilities", ...]
    summary      TEXT        — agent's prose summary
    inline_count INTEGER     — number of inline comments generated
    elapsed_sec  REAL        — total agent processing time
    raw_json     TEXT        — full agent_review_to_dict() output as JSON

USAGE:
  from db.database import init_db, save_review, get_reviews, get_review_stats

  # At server startup:
  init_db()

  # After every agent review:
  save_review(pr_url, pr_number, pr_title, repo, review_dict)

  # API endpoints:
  rows = get_reviews(repo="owner/repo", limit=20, offset=0)
  stats = get_review_stats()
  trend = get_risk_trend(days=30)
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("ai_reviewer.db")

# ─── Connection Management ────────────────────────────────────────────────────
# SQLite connections are NOT thread-safe by default.
# We use threading.local() so each thread gets its own connection object.
# FastAPI runs routes in the same thread pool, so this keeps it safe.

_thread_local = threading.local()
_db_path: str = ""       # set by init_db()
_is_postgres: bool = False


def _get_connection():
    """Return a live DB connection for the current thread."""
    if _is_postgres:
        return _get_postgres_connection()
    return _get_sqlite_connection()


def _get_sqlite_connection() -> sqlite3.Connection:
    """Return (or create) the SQLite connection for this thread."""
    if not hasattr(_thread_local, "conn") or _thread_local.conn is None:
        _thread_local.conn = sqlite3.connect(_db_path, check_same_thread=False)
        _thread_local.conn.row_factory = sqlite3.Row  # dicts instead of tuples
        _thread_local.conn.execute("PRAGMA journal_mode=WAL")  # safe concurrent reads
    return _thread_local.conn


def _get_postgres_connection():
    """Return a psycopg2 connection (for PostgreSQL deployments)."""
    import psycopg2
    import psycopg2.extras
    from config import config
    if not hasattr(_thread_local, "pg_conn") or _thread_local.pg_conn is None or _thread_local.pg_conn.closed:
        _thread_local.pg_conn = psycopg2.connect(config.DATABASE_URL)
        _thread_local.pg_conn.autocommit = False
    return _thread_local.pg_conn


# ─── Schema ────────────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS reviews (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    pr_url       TEXT NOT NULL,
    pr_number    INTEGER,
    pr_title     TEXT,
    repo         TEXT,
    verdict      TEXT,
    risk_level   TEXT,
    confidence   REAL,
    pr_type      TEXT,
    tools_used   TEXT,
    summary      TEXT,
    inline_count INTEGER DEFAULT 0,
    elapsed_sec  REAL,
    raw_json     TEXT,
    feedback     INTEGER DEFAULT 0
);
"""

# Week 8: migration to add feedback to existing databases
_MIGRATE_FEEDBACK_SQL = """
    ALTER TABLE reviews ADD COLUMN feedback INTEGER DEFAULT 0;
"""

_CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_reviews_repo       ON reviews(repo);",
    "CREATE INDEX IF NOT EXISTS idx_reviews_created_at ON reviews(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_reviews_verdict    ON reviews(verdict);",
    "CREATE INDEX IF NOT EXISTS idx_reviews_risk_level ON reviews(risk_level);",
]

_CREATE_TABLE_POSTGRES = """
CREATE TABLE IF NOT EXISTS reviews (
    id           SERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pr_url       TEXT NOT NULL,
    pr_number    INTEGER,
    pr_title     TEXT,
    repo         TEXT,
    verdict      TEXT,
    risk_level   TEXT,
    confidence   REAL,
    pr_type      TEXT,
    tools_used   TEXT,
    summary      TEXT,
    inline_count INTEGER DEFAULT 0,
    elapsed_sec  REAL,
    raw_json     TEXT,
    feedback     INTEGER DEFAULT 0
);
"""


# ─── Public API ───────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create the reviews table if it doesn't exist yet.
    Call once at server startup (in FastAPI lifespan).

    Detects SQLite vs PostgreSQL from DATABASE_URL in config.
    """
    global _db_path, _is_postgres
    from config import config

    url = config.DATABASE_URL or "sqlite:///./reviews.db"

    if url.startswith("postgresql://") or url.startswith("postgres://"):
        _is_postgres = True
        logger.info("DB: Using PostgreSQL")
        conn = _get_postgres_connection()
        cur = conn.cursor()
        cur.execute(_CREATE_TABLE_POSTGRES)
        for idx_sql in _CREATE_INDEX_SQL:
            # Postgres syntax is the same for CREATE INDEX IF NOT EXISTS
            try:
                cur.execute(idx_sql)
            except Exception:
                pass
        conn.commit()
        cur.close()
    else:
        _is_postgres = False
        # Extract file path from sqlite:///./reviews.db or sqlite:////tmp/test.db
        if url.startswith("sqlite:///"):
            rel_path = url[len("sqlite:///"):]
            if os.path.isabs(rel_path):
                # Absolute path (e.g. sqlite:////tmp/test.db or sqlite:///tmp/test.db)
                _db_path = rel_path
            elif rel_path.startswith("./"):
                _db_path = os.path.join(_PROJECT_ROOT, rel_path[2:])
            else:
                _db_path = os.path.join(_PROJECT_ROOT, rel_path)
        else:
            _db_path = os.path.join(_PROJECT_ROOT, "reviews.db")
        logger.info(f"DB: Using SQLite at {_db_path}")
        conn = _get_sqlite_connection()
        conn.execute(_CREATE_TABLE_SQL)
        for idx_sql in _CREATE_INDEX_SQL:
            conn.execute(idx_sql)
        # Week 8: add feedback column to existing databases (idempotent)
        try:
            conn.execute(_MIGRATE_FEEDBACK_SQL)
        except Exception:
            pass  # column already exists — that's fine
        conn.commit()


def save_review(
    pr_url: str,
    pr_number: int,
    pr_title: str,
    repo: str,
    review_dict: dict,
) -> Optional[int]:
    """
    Persist one completed agent review to the database.

    Args:
        pr_url:      Full GitHub PR URL
        pr_number:   Numeric PR id
        pr_title:    PR title from GitHub (can be empty string)
        repo:        "owner/repo" string
        review_dict: Output of agent_review_to_dict() — the full structured review

    Returns:
        The new row's id, or None on failure.

    The function is intentionally non-raising — a DB write failure should
    never crash the review pipeline. We log the error and move on.
    """
    try:
        meta = review_dict.get("agent_meta", {})
        tools = review_dict.get("tools_used", [])
        inline = review_dict.get("inline_comments", [])

        row = {
            "pr_url":       pr_url,
            "pr_number":    pr_number,
            "pr_title":     pr_title or "",
            "repo":         repo or "",
            "verdict":      review_dict.get("overall_verdict", "COMMENT"),
            "risk_level":   review_dict.get("risk_level", "LOW"),
            "confidence":   float(review_dict.get("confidence", 0.0)),
            "pr_type":      meta.get("pr_type", ""),
            "tools_used":   json.dumps(tools),
            "summary":      review_dict.get("summary", ""),
            "inline_count": len(inline),
            "elapsed_sec":  float(meta.get("elapsed_seconds", 0.0)),
            "raw_json":     json.dumps(review_dict),
        }

        conn = _get_connection()

        if _is_postgres:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO reviews
                  (pr_url, pr_number, pr_title, repo, verdict, risk_level,
                   confidence, pr_type, tools_used, summary, inline_count,
                   elapsed_sec, raw_json)
                VALUES
                  (%(pr_url)s, %(pr_number)s, %(pr_title)s, %(repo)s,
                   %(verdict)s, %(risk_level)s, %(confidence)s, %(pr_type)s,
                   %(tools_used)s, %(summary)s, %(inline_count)s,
                   %(elapsed_sec)s, %(raw_json)s)
                RETURNING id
                """,
                row,
            )
            new_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
        else:
            cur = conn.execute(
                """
                INSERT INTO reviews
                  (pr_url, pr_number, pr_title, repo, verdict, risk_level,
                   confidence, pr_type, tools_used, summary, inline_count,
                   elapsed_sec, raw_json)
                VALUES
                  (:pr_url, :pr_number, :pr_title, :repo, :verdict, :risk_level,
                   :confidence, :pr_type, :tools_used, :summary, :inline_count,
                   :elapsed_sec, :raw_json)
                """,
                row,
            )
            conn.commit()
            new_id = cur.lastrowid

        logger.info(f"DB: review saved | id={new_id} pr=#{pr_number} verdict={row['verdict']}")
        return new_id

    except Exception as e:
        logger.error(f"DB: save_review failed (non-fatal): {e}", exc_info=True)
        return None


def get_reviews(
    repo: Optional[str] = None,
    verdict: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """
    Return a paginated list of reviews, newest first.

    Args:
        repo:       Filter by repo ("owner/repo"). None = all repos.
        verdict:    Filter by verdict (REQUEST_CHANGES | APPROVE | COMMENT).
        risk_level: Filter by risk level (LOW | MEDIUM | HIGH).
        limit:      Page size (max 100).
        offset:     Pagination offset.

    Returns:
        List of row dicts (all columns except raw_json for performance).
    """
    try:
        limit = min(limit, 100)
        conditions = []
        params: dict = {"limit": limit, "offset": offset}

        if repo:
            conditions.append("repo = :repo")
            params["repo"] = repo
        if verdict:
            conditions.append("verdict = :verdict")
            params["verdict"] = verdict
        if risk_level:
            conditions.append("risk_level = :risk_level")
            params["risk_level"] = risk_level

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT id, created_at, pr_url, pr_number, pr_title, repo,
                   verdict, risk_level, confidence, pr_type, tools_used,
                   summary, inline_count, elapsed_sec, feedback
            FROM reviews
            {where}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """

        conn = _get_connection()
        if _is_postgres:
            import psycopg2.extras
            # Convert :name params to %(name)s for psycopg2
            pg_sql = sql.replace(":limit", "%(limit)s").replace(":offset", "%(offset)s")
            if repo:       pg_sql = pg_sql.replace(":repo", "%(repo)s")
            if verdict:    pg_sql = pg_sql.replace(":verdict", "%(verdict)s")
            if risk_level: pg_sql = pg_sql.replace(":risk_level", "%(risk_level)s")
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(pg_sql, params)
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
        else:
            cur = conn.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]

        # Parse tools_used JSON string back to list
        for row in rows:
            try:
                row["tools_used"] = json.loads(row["tools_used"] or "[]")
            except Exception:
                row["tools_used"] = []

        return rows

    except Exception as e:
        logger.error(f"DB: get_reviews failed: {e}", exc_info=True)
        return []


def get_review_by_id(review_id: int) -> Optional[dict]:
    """
    Return a single review by its database id, including raw_json.
    Returns None if not found.
    """
    try:
        conn = _get_connection()
        if _is_postgres:
            import psycopg2.extras
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT * FROM reviews WHERE id = %s", (review_id,))
            row = cur.fetchone()
            cur.close()
            if row is None:
                return None
            result = dict(row)
        else:
            cur = conn.execute("SELECT * FROM reviews WHERE id = :id", {"id": review_id})
            row = cur.fetchone()
            if row is None:
                return None
            result = dict(row)

        # Parse JSON fields
        try:
            result["tools_used"] = json.loads(result.get("tools_used") or "[]")
        except Exception:
            result["tools_used"] = []
        try:
            result["raw_json"] = json.loads(result.get("raw_json") or "{}")
        except Exception:
            result["raw_json"] = {}

        return result

    except Exception as e:
        logger.error(f"DB: get_review_by_id failed: {e}", exc_info=True)
        return None


def get_review_stats(repo: Optional[str] = None) -> dict:
    """
    Return aggregate statistics for the dashboard stats row.

    Returns a dict:
    {
        "total": 42,
        "by_verdict": {"REQUEST_CHANGES": 20, "APPROVE": 18, "COMMENT": 4},
        "by_risk":    {"HIGH": 10, "MEDIUM": 15, "LOW": 17},
        "avg_confidence": 0.84,
        "repos": ["owner/repo1", "owner/repo2"],
    }
    """
    try:
        conditions = []
        params: dict = {}
        if repo:
            conditions.append("repo = :repo")
            params["repo"] = repo
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        conn = _get_connection()

        def _run(sql, p=params):
            if _is_postgres:
                pg_sql = sql
                for k in p:
                    pg_sql = pg_sql.replace(f":{k}", f"%({k})s")
                import psycopg2.extras
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute(pg_sql, p)
                rows = [dict(r) for r in cur.fetchall()]
                cur.close()
                return rows
            else:
                cur = conn.execute(sql, p)
                return [dict(r) for r in cur.fetchall()]

        # Total count
        total_rows = _run(f"SELECT COUNT(*) as n FROM reviews {where}")
        total = total_rows[0]["n"] if total_rows else 0

        # By verdict
        verdict_rows = _run(
            f"SELECT verdict, COUNT(*) as n FROM reviews {where} GROUP BY verdict", params
        )
        by_verdict = {r["verdict"]: r["n"] for r in verdict_rows if r["verdict"]}

        # By risk level
        risk_rows = _run(
            f"SELECT risk_level, COUNT(*) as n FROM reviews {where} GROUP BY risk_level", params
        )
        by_risk = {r["risk_level"]: r["n"] for r in risk_rows if r["risk_level"]}

        # Avg confidence
        conf_rows = _run(
            f"SELECT AVG(confidence) as avg_conf FROM reviews {where}", params
        )
        avg_conf = round(float(conf_rows[0]["avg_conf"] or 0.0), 3) if conf_rows else 0.0

        # Distinct repos (only when not filtering by repo already)
        if not repo:
            repo_rows = _run("SELECT DISTINCT repo FROM reviews WHERE repo != '' ORDER BY repo")
            repos = [r["repo"] for r in repo_rows]
        else:
            repos = [repo]

        return {
            "total":          total,
            "by_verdict":     by_verdict,
            "by_risk":        by_risk,
            "avg_confidence": avg_conf,
            "repos":          repos,
        }

    except Exception as e:
        logger.error(f"DB: get_review_stats failed: {e}", exc_info=True)
        return {"total": 0, "by_verdict": {}, "by_risk": {}, "avg_confidence": 0.0, "repos": []}


def get_risk_trend(
    repo: Optional[str] = None,
    days: int = 30,
) -> list[dict]:
    """
    Return daily risk-level counts for the trend chart.

    Returns a list of dicts like:
    [
        {"date": "2026-05-20", "LOW": 3, "MEDIUM": 1, "HIGH": 0},
        {"date": "2026-05-21", "LOW": 0, "MEDIUM": 2, "HIGH": 1},
        ...
    ]
    The list always has exactly `days` entries (missing days filled with zeros),
    so the chart can plot a continuous line.
    """
    try:
        # Build the date range (today going back `days` days)
        today = datetime.utcnow().date()
        date_range = [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]

        # Query grouped counts
        conditions = ["date(created_at) >= :start_date"]
        params: dict = {"start_date": date_range[0]}
        if repo:
            conditions.append("repo = :repo")
            params["repo"] = repo
        where = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT date(created_at) as day, risk_level, COUNT(*) as n
            FROM reviews
            {where}
            GROUP BY day, risk_level
            ORDER BY day
        """

        conn = _get_connection()
        if _is_postgres:
            pg_sql = sql
            for k in params:
                pg_sql = pg_sql.replace(f":{k}", f"%({k})s")
            # Postgres: date() → DATE()
            pg_sql = pg_sql.replace("date(created_at)", "DATE(created_at)")
            pg_sql = pg_sql.replace("date(created_at)", "DATE(created_at)")
            import psycopg2.extras
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(pg_sql, params)
            raw_rows = [dict(r) for r in cur.fetchall()]
            cur.close()
        else:
            cur = conn.execute(sql, params)
            raw_rows = [dict(r) for r in cur.fetchall()]

        # Index raw_rows by (day, risk_level)
        counts: dict[tuple, int] = {}
        for r in raw_rows:
            day_str = str(r["day"])[:10]  # ensure YYYY-MM-DD format
            counts[(day_str, r["risk_level"])] = r["n"]

        # Build complete date series with zeros for missing days
        trend = []
        for day_str in date_range:
            trend.append({
                "date":   day_str,
                "LOW":    counts.get((day_str, "LOW"), 0),
                "MEDIUM": counts.get((day_str, "MEDIUM"), 0),
                "HIGH":   counts.get((day_str, "HIGH"), 0),
            })

        return trend

    except Exception as e:
        logger.error(f"DB: get_risk_trend failed: {e}", exc_info=True)
        return []


def get_total_count(repo: Optional[str] = None) -> int:
    """Return the total number of reviews stored (with optional repo filter)."""
    try:
        params: dict = {}
        where = ""
        if repo:
            where = "WHERE repo = :repo"
            params["repo"] = repo
        conn = _get_connection()
        if _is_postgres:
            sql = f"SELECT COUNT(*) FROM reviews {where}".replace(":repo", "%(repo)s")
            cur = conn.cursor()
            cur.execute(sql, params)
            n = cur.fetchone()[0]
            cur.close()
            return n
        else:
            cur = conn.execute(f"SELECT COUNT(*) as n FROM reviews {where}", params)
            return cur.fetchone()["n"]
    except Exception as e:
        logger.error(f"DB: get_total_count failed: {e}", exc_info=True)
        return 0


def set_review_feedback(review_id: int, score: int) -> bool:
    """
    Save developer feedback (thumbs up / down) for a review.

    Args:
        review_id: The review's database id.
        score:      1  = thumbs up (👍)
                   -1  = thumbs down (👎)
                    0  = clear feedback

    Returns:
        True if the row was found and updated, False otherwise.

    Design note: feedback is stored on the reviews row itself (not a
    separate table) for simplicity. One review, one feedback signal.
    If you later want per-reviewer granularity, add a separate table.
    """
    if score not in (1, -1, 0):
        raise ValueError(f"score must be 1, -1, or 0 — got {score}")
    try:
        conn = _get_connection()
        if _is_postgres:
            cur = conn.cursor()
            cur.execute(
                "UPDATE reviews SET feedback = %s WHERE id = %s",
                (score, review_id)
            )
            updated = cur.rowcount
            conn.commit()
            cur.close()
        else:
            cur = conn.execute(
                "UPDATE reviews SET feedback = :score WHERE id = :id",
                {"score": score, "id": review_id}
            )
            conn.commit()
            updated = cur.rowcount

        if updated:
            label = {1: "👍 thumbs up", -1: "👎 thumbs down", 0: "cleared"}[score]
            logger.info(f"DB: feedback {label} | review_id={review_id}")
            return True
        else:
            logger.warning(f"DB: set_review_feedback — review {review_id} not found")
            return False

    except Exception as e:
        logger.error(f"DB: set_review_feedback failed: {e}", exc_info=True)
        return False
