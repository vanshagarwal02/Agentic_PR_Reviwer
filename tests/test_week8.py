"""
test_week8.py — Tests for Week 8: Evaluation Framework & Feedback Loop.

Tests:
  1. Benchmark dataset structure validation
  2. Metric computation helpers (precision, recall, F1)
  3. set_review_feedback() DB function
  4. POST /api/reviews/{id}/feedback FastAPI endpoint
  5. GET /api/eval/results FastAPI endpoint
  6. evaluate.py dry-run mode (no Gemini calls)
"""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

BENCHMARK_PATH = os.path.join(_ROOT, "data", "eval_benchmark.json")


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mem_db(tmp_path, monkeypatch):
    """SQLite DB in a temp file — same pattern as test_week7."""
    import db.database as db_mod
    import config as cfg_mod

    db_file = str(tmp_path / "test_w8.db")
    sqlite_url = f"sqlite:///{db_file}"

    monkeypatch.setattr(cfg_mod.config, "DATABASE_URL", sqlite_url)
    monkeypatch.setattr(db_mod, "_db_path", db_file)
    monkeypatch.setattr(db_mod, "_is_postgres", False)

    if hasattr(db_mod._thread_local, "conn") and db_mod._thread_local.conn:
        try:
            db_mod._thread_local.conn.close()
        except Exception:
            pass
        db_mod._thread_local.conn = None

    db_mod.init_db()
    yield db_mod

    if hasattr(db_mod._thread_local, "conn") and db_mod._thread_local.conn:
        db_mod._thread_local.conn.close()
        db_mod._thread_local.conn = None


SAMPLE_REVIEW = {
    "overall_verdict": "REQUEST_CHANGES",
    "risk_level": "HIGH",
    "summary": "SQL injection found.",
    "inline_comments": [],
    "tools_used": ["scan_for_vulnerabilities"],
    "confidence": 0.91,
    "agent_meta": {
        "pr_type": "feature",
        "reasoning": "",
        "gemini_calls": 2,
        "elapsed_seconds": 18.5,
        "error": None,
    },
}


# ─── 1. Benchmark Dataset Validation ──────────────────────────────────────────

class TestBenchmarkDataset:
    def test_benchmark_file_exists(self):
        assert os.path.exists(BENCHMARK_PATH), \
            f"Benchmark not found at {BENCHMARK_PATH}"

    def test_benchmark_is_valid_json(self):
        with open(BENCHMARK_PATH) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_benchmark_has_20_samples(self):
        with open(BENCHMARK_PATH) as f:
            data = json.load(f)
        assert len(data) == 20, f"Expected 20 samples, got {len(data)}"

    def test_benchmark_has_10_vulnerable(self):
        with open(BENCHMARK_PATH) as f:
            data = json.load(f)
        vuln = [s for s in data if s["expected_verdict"] == "REQUEST_CHANGES"]
        assert len(vuln) == 10

    def test_benchmark_has_10_safe(self):
        with open(BENCHMARK_PATH) as f:
            data = json.load(f)
        safe = [s for s in data if s["expected_verdict"] == "APPROVE"]
        assert len(safe) == 10

    def test_all_samples_have_required_fields(self):
        required = {"id", "function_name", "file_path", "function_source",
                    "expected_verdict", "expected_risk", "vulnerability_type"}
        with open(BENCHMARK_PATH) as f:
            data = json.load(f)
        for sample in data:
            missing = required - set(sample.keys())
            assert not missing, f"Sample {sample.get('id')} missing: {missing}"

    def test_all_ids_unique(self):
        with open(BENCHMARK_PATH) as f:
            data = json.load(f)
        ids = [s["id"] for s in data]
        assert len(ids) == len(set(ids)), "Duplicate IDs in benchmark"

    def test_vulnerability_types_covered(self):
        with open(BENCHMARK_PATH) as f:
            data = json.load(f)
        types = {s["vulnerability_type"] for s in data if s["expected_verdict"] == "REQUEST_CHANGES"}
        expected_types = {"SQL_INJECTION", "HARDCODED_SECRET", "PATH_TRAVERSAL",
                          "CMD_INJECTION", "INSECURE_DESERIALIZATION"}
        assert expected_types == types


# ─── 2. Metric Computation ────────────────────────────────────────────────────

class TestMetricComputation:
    """Test the _compute_metrics helper from evaluate.py."""

    @pytest.fixture
    def compute_metrics(self):
        from scripts.evaluate import _compute_metrics
        return _compute_metrics

    def _make_results(self, tp=3, fp=1, fn=1, tn=5):
        results = []
        for _ in range(tp):
            results.append({"expected_verdict": "REQUEST_CHANGES",
                            "actual_verdict": "REQUEST_CHANGES",
                            "expected_risk": "HIGH", "actual_risk": "HIGH",
                            "vulnerability_type": "SQL_INJECTION"})
        for _ in range(fp):
            results.append({"expected_verdict": "APPROVE",
                            "actual_verdict": "REQUEST_CHANGES",
                            "expected_risk": "LOW", "actual_risk": "MEDIUM",
                            "vulnerability_type": "NONE"})
        for _ in range(fn):
            results.append({"expected_verdict": "REQUEST_CHANGES",
                            "actual_verdict": "APPROVE",
                            "expected_risk": "HIGH", "actual_risk": "LOW",
                            "vulnerability_type": "CMD_INJECTION"})
        for _ in range(tn):
            results.append({"expected_verdict": "APPROVE",
                            "actual_verdict": "APPROVE",
                            "expected_risk": "LOW", "actual_risk": "LOW",
                            "vulnerability_type": "NONE"})
        return results

    def test_precision_calculation(self, compute_metrics):
        results = self._make_results(tp=3, fp=1, fn=0, tn=5)
        m = compute_metrics(results)
        # precision = tp / (tp + fp) = 3/4 = 0.75
        assert abs(m["precision"] - 0.75) < 0.01

    def test_recall_calculation(self, compute_metrics):
        results = self._make_results(tp=3, fp=0, fn=1, tn=5)
        m = compute_metrics(results)
        # recall = tp / (tp + fn) = 3/4 = 0.75
        assert abs(m["recall"] - 0.75) < 0.01

    def test_f1_calculation(self, compute_metrics):
        results = self._make_results(tp=3, fp=1, fn=1, tn=5)
        m = compute_metrics(results)
        p = 3 / 4  # precision
        r = 3 / 4  # recall
        expected_f1 = 2 * p * r / (p + r)
        assert abs(m["f1"] - expected_f1) < 0.01

    def test_perfect_precision_when_no_fp(self, compute_metrics):
        results = self._make_results(tp=5, fp=0, fn=0, tn=5)
        m = compute_metrics(results)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_counts_correct(self, compute_metrics):
        results = self._make_results(tp=3, fp=1, fn=1, tn=5)
        m = compute_metrics(results)
        assert m["tp"] == 3
        assert m["fp"] == 1
        assert m["fn"] == 1
        assert m["tn"] == 5
        assert m["total"] == 10
        assert m["vulnerable"] == 4
        assert m["safe"] == 6

    def test_per_category_recall(self, compute_metrics):
        results = [
            {"expected_verdict": "REQUEST_CHANGES", "actual_verdict": "REQUEST_CHANGES",
             "expected_risk": "HIGH", "actual_risk": "HIGH", "vulnerability_type": "SQL_INJECTION"},
            {"expected_verdict": "REQUEST_CHANGES", "actual_verdict": "REQUEST_CHANGES",
             "expected_risk": "HIGH", "actual_risk": "HIGH", "vulnerability_type": "SQL_INJECTION"},
            {"expected_verdict": "REQUEST_CHANGES", "actual_verdict": "APPROVE",
             "expected_risk": "HIGH", "actual_risk": "LOW", "vulnerability_type": "CMD_INJECTION"},
        ]
        m = compute_metrics(results)
        assert m["per_category"]["SQL_INJECTION"]["recall"] == 1.0
        assert m["per_category"]["CMD_INJECTION"]["recall"] == 0.0


# ─── 3. set_review_feedback() ─────────────────────────────────────────────────

class TestSetReviewFeedback:
    def _seed_review(self, mem_db):
        return mem_db.save_review(
            "https://github.com/o/r/pull/1", 1, "Test", "o/r", SAMPLE_REVIEW
        )

    def test_thumbs_up_stored(self, mem_db):
        row_id = self._seed_review(mem_db)
        result = mem_db.set_review_feedback(row_id, 1)
        assert result is True
        conn = mem_db._get_sqlite_connection()
        row = conn.execute("SELECT feedback FROM reviews WHERE id=?", (row_id,)).fetchone()
        assert row["feedback"] == 1

    def test_thumbs_down_stored(self, mem_db):
        row_id = self._seed_review(mem_db)
        mem_db.set_review_feedback(row_id, -1)
        conn = mem_db._get_sqlite_connection()
        row = conn.execute("SELECT feedback FROM reviews WHERE id=?", (row_id,)).fetchone()
        assert row["feedback"] == -1

    def test_clear_feedback(self, mem_db):
        row_id = self._seed_review(mem_db)
        mem_db.set_review_feedback(row_id, 1)
        mem_db.set_review_feedback(row_id, 0)
        conn = mem_db._get_sqlite_connection()
        row = conn.execute("SELECT feedback FROM reviews WHERE id=?", (row_id,)).fetchone()
        assert row["feedback"] == 0

    def test_update_replaces_not_inserts(self, mem_db):
        """Setting feedback twice should update the same row, not add a new one."""
        row_id = self._seed_review(mem_db)
        mem_db.set_review_feedback(row_id, 1)
        mem_db.set_review_feedback(row_id, -1)
        conn = mem_db._get_sqlite_connection()
        count = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        assert count == 1  # still one row

    def test_returns_false_for_missing_id(self, mem_db):
        result = mem_db.set_review_feedback(99999, 1)
        assert result is False

    def test_invalid_score_raises(self, mem_db):
        row_id = self._seed_review(mem_db)
        with pytest.raises(ValueError):
            mem_db.set_review_feedback(row_id, 42)

    def test_feedback_included_in_get_reviews(self, mem_db):
        row_id = self._seed_review(mem_db)
        mem_db.set_review_feedback(row_id, 1)
        rows = mem_db.get_reviews()
        assert rows[0]["feedback"] == 1

    def test_feedback_column_exists_after_init(self, mem_db):
        """init_db() migration should add feedback column to existing DB."""
        conn = mem_db._get_sqlite_connection()
        cur = conn.execute("PRAGMA table_info(reviews)")
        columns = {row["name"] for row in cur.fetchall()}
        assert "feedback" in columns


# ─── 4. Feedback API Endpoint ─────────────────────────────────────────────────

class TestFeedbackEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        with patch("db.database.init_db"):
            with patch("tools.vulnerability_scanner.ensure_index_exists"):
                import main as app_module
                return TestClient(app_module.app)

    @patch("main.set_review_feedback", return_value=True)
    def test_thumbs_up_returns_200(self, mock_fb, client):
        resp = client.post("/api/reviews/1/feedback", json={"score": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["feedback"] == 1
        mock_fb.assert_called_once_with(1, 1)

    @patch("main.set_review_feedback", return_value=True)
    def test_thumbs_down_returns_200(self, mock_fb, client):
        resp = client.post("/api/reviews/2/feedback", json={"score": -1})
        assert resp.status_code == 200
        assert resp.json()["feedback"] == -1

    @patch("main.set_review_feedback", return_value=True)
    def test_clear_feedback_returns_200(self, mock_fb, client):
        resp = client.post("/api/reviews/3/feedback", json={"score": 0})
        assert resp.status_code == 200

    @patch("main.set_review_feedback", return_value=False)
    def test_returns_404_for_missing_review(self, mock_fb, client):
        resp = client.post("/api/reviews/99999/feedback", json={"score": 1})
        assert resp.status_code == 404

    def test_invalid_score_returns_400(self, client):
        resp = client.post("/api/reviews/1/feedback", json={"score": 99})
        assert resp.status_code == 400

    def test_missing_body_returns_422(self, client):
        resp = client.post("/api/reviews/1/feedback", json={})
        assert resp.status_code == 422


# ─── 5. Eval Results Endpoint ─────────────────────────────────────────────────

class TestEvalResultsEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        with patch("db.database.init_db"):
            with patch("tools.vulnerability_scanner.ensure_index_exists"):
                import main as app_module
                return TestClient(app_module.app)

    def test_returns_404_when_no_results(self, client, tmp_path, monkeypatch):
        import config as cfg_mod
        monkeypatch.setattr(cfg_mod.config, "BASE_DIR", str(tmp_path))
        resp = client.get("/api/eval/results")
        assert resp.status_code == 404

    def test_returns_results_when_file_exists(self, client, tmp_path, monkeypatch):
        import config as cfg_mod
        monkeypatch.setattr(cfg_mod.config, "BASE_DIR", str(tmp_path))
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results = {"metrics": {"precision": 0.95, "f1": 0.95}, "results": []}
        with open(data_dir / "eval_results.json", "w") as f:
            json.dump(results, f)
        resp = client.get("/api/eval/results")
        assert resp.status_code == 200
        assert resp.json()["metrics"]["f1"] == 0.95


# ─── 6. Evaluate.py Dry-run ───────────────────────────────────────────────────

class TestEvaluateDryRun:
    def test_dry_run_validates_dataset(self):
        from scripts.evaluate import run_evaluation
        result = run_evaluation(BENCHMARK_PATH, dry_run=True)
        assert result.get("dry_run") is True

    def test_dry_run_does_not_call_agent(self):
        """In dry-run mode, the agent (run_agent_review) is never imported or called."""
        from scripts.evaluate import run_evaluation
        # Patch at the agent module level — it would be imported lazily inside run_evaluation
        with patch("agent.react_agent.run_agent_review") as mock_agent:
            run_evaluation(BENCHMARK_PATH, dry_run=True)
        mock_agent.assert_not_called()

    def test_dry_run_raises_on_invalid_dataset(self, tmp_path):
        from scripts.evaluate import run_evaluation
        bad_path = str(tmp_path / "bad.json")
        with open(bad_path, "w") as f:
            json.dump([{"id": "x"}], f)  # missing required fields
        with pytest.raises(ValueError, match="missing fields"):
            run_evaluation(bad_path, dry_run=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
