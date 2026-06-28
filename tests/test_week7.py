"""
test_week7.py — Tests for the Week 7 persistence layer and API endpoints.

Test strategy:
  - Use an in-memory SQLite DB (":memory:") to keep tests fast and isolated.
  - Monkeypatch _db_path so database.py writes to the in-memory connection.
  - Test each public function: init_db, save_review, get_reviews, get_review_by_id,
    get_review_stats, get_risk_trend, get_total_count.
  - Integration tests using FastAPI TestClient for the /api/reviews/* endpoints.
"""

import json
import os
import sys
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_REVIEW_DICT = {
    "overall_verdict": "REQUEST_CHANGES",
    "risk_level": "HIGH",
    "summary": "SQL injection found in get_user.",
    "inline_comments": [
        {"file": "db.py", "line": 5, "severity": "CRITICAL",
         "category": "SECURITY", "comment": "SQL injection", "suggestion": "Use parameterised queries"},
    ],
    "tools_used": ["scan_for_vulnerabilities"],
    "confidence": 0.91,
    "agent_meta": {
        "pr_type": "feature",
        "reasoning": "DB code changed",
        "gemini_calls": 2,
        "elapsed_seconds": 18.5,
        "error": None,
    },
}

APPROVE_REVIEW_DICT = {
    "overall_verdict": "APPROVE",
    "risk_level": "LOW",
    "summary": "No issues found.",
    "inline_comments": [],
    "tools_used": [],
    "confidence": 0.80,
    "agent_meta": {
        "pr_type": "style",
        "reasoning": "",
        "gemini_calls": 2,
        "elapsed_seconds": 12.0,
        "error": None,
    },
}


@pytest.fixture
def mem_db(tmp_path, monkeypatch):
    """
    Create a fresh SQLite DB in a temp file for each test.
    Monkeypatches config.DATABASE_URL → sqlite so init_db() takes the SQLite
    branch regardless of what is set in .env.
    Returns the db.database module so tests can call its functions directly.
    """
    import db.database as db_mod
    import config as cfg_mod

    db_file = str(tmp_path / "test_reviews.db")
    sqlite_url = f"sqlite:///{db_file}"

    # Force SQLite in both the config object AND the db module globals
    monkeypatch.setattr(cfg_mod.config, "DATABASE_URL", sqlite_url)
    monkeypatch.setattr(db_mod, "_db_path", db_file)
    monkeypatch.setattr(db_mod, "_is_postgres", False)

    # Clear any existing thread-local connections from prior tests
    if hasattr(db_mod._thread_local, "conn") and db_mod._thread_local.conn:
        try:
            db_mod._thread_local.conn.close()
        except Exception:
            pass
        db_mod._thread_local.conn = None

    db_mod.init_db()
    yield db_mod

    # Cleanup
    if hasattr(db_mod._thread_local, "conn") and db_mod._thread_local.conn:
        db_mod._thread_local.conn.close()
        db_mod._thread_local.conn = None


# ─── init_db Tests ────────────────────────────────────────────────────────────

class TestInitDb:
    def test_creates_reviews_table(self, mem_db):
        conn = mem_db._get_sqlite_connection()
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='reviews'"
        )
        assert cur.fetchone() is not None, "reviews table should exist after init_db()"

    def test_creates_indexes(self, mem_db):
        conn = mem_db._get_sqlite_connection()
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        index_names = {row[0] for row in cur.fetchall()}
        assert "idx_reviews_repo" in index_names
        assert "idx_reviews_created_at" in index_names

    def test_idempotent(self, mem_db):
        """Calling init_db() twice should not raise or duplicate tables."""
        mem_db.init_db()  # second call
        conn = mem_db._get_sqlite_connection()
        cur = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='reviews'"
        )
        assert cur.fetchone()[0] == 1


# ─── save_review Tests ────────────────────────────────────────────────────────

class TestSaveReview:
    def test_returns_integer_id(self, mem_db):
        row_id = mem_db.save_review(
            pr_url="https://github.com/o/r/pull/1",
            pr_number=1,
            pr_title="Test PR",
            repo="o/r",
            review_dict=SAMPLE_REVIEW_DICT,
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_persists_correct_fields(self, mem_db):
        mem_db.save_review(
            pr_url="https://github.com/o/r/pull/2",
            pr_number=2,
            pr_title="Bug fix",
            repo="o/r",
            review_dict=SAMPLE_REVIEW_DICT,
        )
        conn = mem_db._get_sqlite_connection()
        row = conn.execute("SELECT * FROM reviews WHERE pr_number=2").fetchone()
        assert row["verdict"] == "REQUEST_CHANGES"
        assert row["risk_level"] == "HIGH"
        assert abs(row["confidence"] - 0.91) < 0.001
        assert row["pr_title"] == "Bug fix"
        assert row["repo"] == "o/r"
        assert row["inline_count"] == 1
        assert abs(row["elapsed_sec"] - 18.5) < 0.001

    def test_tools_used_stored_as_json(self, mem_db):
        mem_db.save_review("u", 3, "t", "o/r", SAMPLE_REVIEW_DICT)
        conn = mem_db._get_sqlite_connection()
        row = conn.execute("SELECT tools_used FROM reviews WHERE pr_number=3").fetchone()
        tools = json.loads(row["tools_used"])
        assert "scan_for_vulnerabilities" in tools

    def test_raw_json_stored(self, mem_db):
        mem_db.save_review("u", 4, "t", "o/r", SAMPLE_REVIEW_DICT)
        conn = mem_db._get_sqlite_connection()
        row = conn.execute("SELECT raw_json FROM reviews WHERE pr_number=4").fetchone()
        parsed = json.loads(row["raw_json"])
        assert parsed["overall_verdict"] == "REQUEST_CHANGES"

    def test_does_not_raise_on_bad_data(self, mem_db):
        """save_review should never raise — it logs and returns None."""
        result = mem_db.save_review("url", None, None, None, {})
        # Should return either None or an int — not raise
        assert result is None or isinstance(result, int)

    def test_multiple_saves_increment_ids(self, mem_db):
        id1 = mem_db.save_review("u1", 10, "t1", "o/r", APPROVE_REVIEW_DICT)
        id2 = mem_db.save_review("u2", 11, "t2", "o/r", SAMPLE_REVIEW_DICT)
        assert id2 > id1


# ─── get_reviews Tests ────────────────────────────────────────────────────────

class TestGetReviews:
    def _seed(self, mem_db, n=5):
        for i in range(n):
            mem_db.save_review(
                pr_url=f"https://github.com/o/r/pull/{i}",
                pr_number=i,
                pr_title=f"PR {i}",
                repo="o/r" if i % 2 == 0 else "o/other",
                review_dict=SAMPLE_REVIEW_DICT if i % 3 != 0 else APPROVE_REVIEW_DICT,
            )

    def test_returns_list(self, mem_db):
        self._seed(mem_db)
        rows = mem_db.get_reviews()
        assert isinstance(rows, list)

    def test_pagination_limit(self, mem_db):
        self._seed(mem_db, 10)
        rows = mem_db.get_reviews(limit=3)
        assert len(rows) <= 3

    def test_pagination_offset(self, mem_db):
        self._seed(mem_db, 6)
        all_rows = mem_db.get_reviews(limit=6)
        page2    = mem_db.get_reviews(limit=3, offset=3)
        assert all(r not in all_rows[:3] or True for r in page2)  # basic smoke test
        # Specifically: row ids should be different
        ids_all = [r["id"] for r in all_rows]
        ids_p2  = [r["id"] for r in page2]
        assert ids_p2 == ids_all[3:6]

    def test_filter_by_repo(self, mem_db):
        self._seed(mem_db, 6)
        rows = mem_db.get_reviews(repo="o/other")
        assert all(r["repo"] == "o/other" for r in rows)

    def test_filter_by_verdict(self, mem_db):
        self._seed(mem_db, 6)
        rows = mem_db.get_reviews(verdict="APPROVE")
        assert all(r["verdict"] == "APPROVE" for r in rows)

    def test_filter_by_risk(self, mem_db):
        self._seed(mem_db, 6)
        rows = mem_db.get_reviews(risk_level="HIGH")
        assert all(r["risk_level"] == "HIGH" for r in rows)

    def test_tools_used_is_list(self, mem_db):
        self._seed(mem_db, 2)
        rows = mem_db.get_reviews()
        for r in rows:
            assert isinstance(r["tools_used"], list)

    def test_max_limit_capped_at_100(self, mem_db):
        self._seed(mem_db, 5)
        # Passing 999 should be silently capped to 100
        rows = mem_db.get_reviews(limit=999)
        assert len(rows) <= 100


# ─── get_review_by_id Tests ───────────────────────────────────────────────────

class TestGetReviewById:
    def test_returns_none_for_missing_id(self, mem_db):
        result = mem_db.get_review_by_id(99999)
        assert result is None

    def test_returns_dict_for_valid_id(self, mem_db):
        row_id = mem_db.save_review("u", 1, "t", "o/r", SAMPLE_REVIEW_DICT)
        result = mem_db.get_review_by_id(row_id)
        assert result is not None
        assert isinstance(result, dict)

    def test_raw_json_is_parsed_dict(self, mem_db):
        row_id = mem_db.save_review("u", 1, "t", "o/r", SAMPLE_REVIEW_DICT)
        result = mem_db.get_review_by_id(row_id)
        assert isinstance(result["raw_json"], dict)
        assert "overall_verdict" in result["raw_json"]

    def test_tools_used_is_list(self, mem_db):
        row_id = mem_db.save_review("u", 1, "t", "o/r", SAMPLE_REVIEW_DICT)
        result = mem_db.get_review_by_id(row_id)
        assert isinstance(result["tools_used"], list)


# ─── get_review_stats Tests ───────────────────────────────────────────────────

class TestGetReviewStats:
    def test_empty_db_returns_zeros(self, mem_db):
        stats = mem_db.get_review_stats()
        assert stats["total"] == 0
        assert stats["by_verdict"] == {}
        assert stats["by_risk"] == {}

    def test_total_count_correct(self, mem_db):
        for i in range(4):
            mem_db.save_review(f"u{i}", i, "t", "o/r", SAMPLE_REVIEW_DICT)
        stats = mem_db.get_review_stats()
        assert stats["total"] == 4

    def test_by_verdict_correct(self, mem_db):
        mem_db.save_review("u1", 1, "t", "o/r", SAMPLE_REVIEW_DICT)  # REQUEST_CHANGES
        mem_db.save_review("u2", 2, "t", "o/r", APPROVE_REVIEW_DICT)   # APPROVE
        mem_db.save_review("u3", 3, "t", "o/r", APPROVE_REVIEW_DICT)   # APPROVE
        stats = mem_db.get_review_stats()
        assert stats["by_verdict"].get("REQUEST_CHANGES") == 1
        assert stats["by_verdict"].get("APPROVE") == 2

    def test_avg_confidence_correct(self, mem_db):
        mem_db.save_review("u1", 1, "t", "o/r", SAMPLE_REVIEW_DICT)   # conf=0.91
        mem_db.save_review("u2", 2, "t", "o/r", APPROVE_REVIEW_DICT)   # conf=0.80
        stats = mem_db.get_review_stats()
        expected = round((0.91 + 0.80) / 2, 3)
        assert abs(stats["avg_confidence"] - expected) < 0.01

    def test_repo_filter(self, mem_db):
        mem_db.save_review("u1", 1, "t", "owner/repo1", SAMPLE_REVIEW_DICT)
        mem_db.save_review("u2", 2, "t", "owner/repo2", APPROVE_REVIEW_DICT)
        stats = mem_db.get_review_stats(repo="owner/repo1")
        assert stats["total"] == 1

    def test_repos_list_contains_all_repos(self, mem_db):
        mem_db.save_review("u1", 1, "t", "owner/repo1", SAMPLE_REVIEW_DICT)
        mem_db.save_review("u2", 2, "t", "owner/repo2", APPROVE_REVIEW_DICT)
        stats = mem_db.get_review_stats()
        assert "owner/repo1" in stats["repos"]
        assert "owner/repo2" in stats["repos"]


# ─── get_risk_trend Tests ─────────────────────────────────────────────────────

class TestGetRiskTrend:
    def test_returns_correct_number_of_days(self, mem_db):
        trend = mem_db.get_risk_trend(days=14)
        assert len(trend) == 14

    def test_each_entry_has_required_keys(self, mem_db):
        trend = mem_db.get_risk_trend(days=7)
        for entry in trend:
            assert "date" in entry
            assert "LOW" in entry
            assert "MEDIUM" in entry
            assert "HIGH" in entry

    def test_empty_days_filled_with_zeros(self, mem_db):
        """Days with no reviews should have 0 for all risk levels."""
        trend = mem_db.get_risk_trend(days=30)
        # All should be 0 since DB is empty
        for entry in trend:
            assert entry["LOW"] == 0
            assert entry["MEDIUM"] == 0
            assert entry["HIGH"] == 0

    def test_counts_reviews_on_correct_day(self, mem_db):
        mem_db.save_review("u1", 1, "t", "o/r", SAMPLE_REVIEW_DICT)  # HIGH
        trend = mem_db.get_risk_trend(days=7)
        today_str = datetime.utcnow().date().isoformat()
        today_entry = next((e for e in trend if e["date"] == today_str), None)
        assert today_entry is not None
        assert today_entry["HIGH"] >= 1

    def test_dates_are_ordered_ascending(self, mem_db):
        trend = mem_db.get_risk_trend(days=10)
        dates = [e["date"] for e in trend]
        assert dates == sorted(dates)


# ─── get_total_count Tests ────────────────────────────────────────────────────

class TestGetTotalCount:
    def test_zero_when_empty(self, mem_db):
        assert mem_db.get_total_count() == 0

    def test_counts_all_rows(self, mem_db):
        for i in range(3):
            mem_db.save_review(f"u{i}", i, "t", "o/r", APPROVE_REVIEW_DICT)
        assert mem_db.get_total_count() == 3

    def test_filters_by_repo(self, mem_db):
        mem_db.save_review("u1", 1, "t", "owner/repoA", APPROVE_REVIEW_DICT)
        mem_db.save_review("u2", 2, "t", "owner/repoB", APPROVE_REVIEW_DICT)
        assert mem_db.get_total_count(repo="owner/repoA") == 1


# ─── FastAPI Integration Tests ────────────────────────────────────────────────

class TestApiEndpoints:
    """
    These tests use FastAPI's TestClient to call the /api/reviews/* endpoints.
    We mock out the db.database functions so we don't need a real DB.
    """

    @pytest.fixture
    def client(self):
        """Build a TestClient with all DB calls mocked out."""
        from fastapi.testclient import TestClient
        with patch("db.database.init_db"):
            with patch("tools.vulnerability_scanner.ensure_index_exists"):
                import main as app_module
                return TestClient(app_module.app)

    @patch("main.get_reviews", return_value=[])
    @patch("main.get_total_count", return_value=0)
    def test_api_reviews_returns_200(self, mock_count, mock_reviews, client):
        resp = client.get("/api/reviews")
        assert resp.status_code == 200
        data = resp.json()
        assert "reviews" in data
        assert "total" in data

    @patch("main.get_review_stats", return_value={
        "total": 5,
        "by_verdict": {"APPROVE": 3, "REQUEST_CHANGES": 2},
        "by_risk": {"LOW": 3, "HIGH": 2},
        "avg_confidence": 0.85,
        "repos": ["o/r"],
    })
    def test_api_stats_returns_stats(self, mock_stats, client):
        resp = client.get("/api/reviews/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert "by_verdict" in data

    @patch("main.get_risk_trend", return_value=[
        {"date": "2026-05-01", "LOW": 1, "MEDIUM": 0, "HIGH": 0}
    ])
    def test_api_trend_returns_list(self, mock_trend, client):
        resp = client.get("/api/reviews/trend?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert "date" in data[0]

    @patch("main.get_review_by_id", return_value=None)
    def test_api_review_by_id_404_when_missing(self, mock_get, client):
        resp = client.get("/api/reviews/99999")
        assert resp.status_code == 404

    @patch("main.get_review_by_id", return_value={
        "id": 1, "pr_number": 7, "pr_title": "Test PR",
        "verdict": "APPROVE", "risk_level": "LOW",
        "confidence": 0.9, "repo": "o/r", "pr_url": "https://github.com/o/r/pull/7",
        "created_at": "2026-05-28T12:00:00", "pr_type": "style",
        "tools_used": [], "summary": "All good.", "inline_count": 0,
        "elapsed_sec": 10.0, "raw_json": {},
    })
    def test_api_review_by_id_returns_detail(self, mock_get, client):
        resp = client.get("/api/reviews/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pr_number"] == 7
        assert data["verdict"] == "APPROVE"

    @patch("main.get_reviews", return_value=[])
    @patch("main.get_total_count", return_value=0)
    def test_api_reviews_accepts_filter_params(self, mock_count, mock_reviews, client):
        resp = client.get("/api/reviews?verdict=APPROVE&risk_level=LOW&limit=10&offset=0")
        assert resp.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
