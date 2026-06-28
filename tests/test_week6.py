"""
test_week6.py — Tests for GitHub review posting (Week 6)

Test strategy:
  - Mock all HTTP calls (no real GitHub API calls needed)
  - Test URL parsing, body formatting, diff-line filtering, fallback logic
  - Integration test: end-to-end with fully mocked requests
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch, call
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tools.github_poster import (
    _parse_pr_url,
    _build_review_body,
    _build_inline_body,
    post_review_to_github,
    PostResult,
    SEVERITY_EMOJI,
    VERDICT_EMOJI,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_REVIEW = {
    "overall_verdict": "REQUEST_CHANGES",
    "risk_level": "HIGH",
    "summary": "SQL injection found in get_user function.",
    "inline_comments": [
        {
            "file": "db.py",
            "line": 5,
            "severity": "CRITICAL",
            "category": "SECURITY",
            "comment": "SQL injection via string concatenation",
            "suggestion": "cursor.execute(query, (username,))",
        },
        {
            "file": "auth.py",
            "line": 20,
            "severity": "HIGH",
            "category": "SECURITY",
            "comment": "Hardcoded secret key",
            "suggestion": "Use os.environ['SECRET_KEY']",
        },
    ],
    "tools_used": ["scan_for_vulnerabilities", "check_vulnerability_patterns"],
    "confidence": 0.91,
    "agent_meta": {
        "pr_type": "feature",
        "reasoning": "PR touches DB code",
        "gemini_calls": 2,
        "elapsed_seconds": 18.5,
        "error": None,
    },
}

EMPTY_REVIEW = {
    "overall_verdict": "APPROVE",
    "risk_level": "LOW",
    "summary": "No issues found.",
    "inline_comments": [],
    "tools_used": [],
    "confidence": 0.8,
    "agent_meta": {"pr_type": "style", "reasoning": "", "gemini_calls": 2,
                   "elapsed_seconds": 10.0, "error": None},
}


# ─── URL Parser Tests ─────────────────────────────────────────────────────────

class TestPrUrlParser:
    def test_parses_standard_url(self):
        owner, repo, num = _parse_pr_url("https://github.com/owner/myrepo/pull/42")
        assert owner == "owner"
        assert repo == "myrepo"
        assert num == 42

    def test_parses_url_with_files_suffix(self):
        owner, repo, num = _parse_pr_url(
            "https://github.com/apeksharahulchandak/ai-reviewer-test/pull/13/files"
        )
        assert owner == "apeksharahulchandak"
        assert repo == "ai-reviewer-test"
        assert num == 13

    def test_parses_real_repo_url(self):
        owner, repo, num = _parse_pr_url(
            "https://github.com/Apekshachandak/ai-reviewer-test/pull/11"
        )
        assert owner == "Apekshachandak"
        assert num == 11

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError):
            _parse_pr_url("https://gitlab.com/owner/repo/merge_requests/1")

    def test_non_github_url_raises(self):
        with pytest.raises(ValueError):
            _parse_pr_url("https://example.com/not/a/pr")


# ─── Review Body Formatter Tests ──────────────────────────────────────────────

class TestReviewBodyFormatter:
    def test_body_contains_verdict(self):
        body = _build_review_body(SAMPLE_REVIEW, [])
        assert "REQUEST_CHANGES" in body or "Request Changes" in body

    def test_body_contains_summary(self):
        body = _build_review_body(SAMPLE_REVIEW, [])
        assert "SQL injection found" in body

    def test_body_contains_risk_level(self):
        body = _build_review_body(SAMPLE_REVIEW, [])
        assert "HIGH" in body

    def test_body_contains_tools(self):
        body = _build_review_body(SAMPLE_REVIEW, [])
        assert "scan_for_vulnerabilities" in body

    def test_body_contains_confidence(self):
        body = _build_review_body(SAMPLE_REVIEW, [])
        assert "91%" in body

    def test_body_includes_fallback_comments(self):
        fallbacks = [SAMPLE_REVIEW["inline_comments"][0]]
        body = _build_review_body(SAMPLE_REVIEW, fallbacks)
        assert "Additional Findings" in body
        assert "db.py" in body

    def test_body_no_fallback_section_when_empty(self):
        body = _build_review_body(SAMPLE_REVIEW, [])
        assert "Additional Findings" not in body

    def test_body_has_branding(self):
        body = _build_review_body(SAMPLE_REVIEW, [])
        assert "AI Code Reviewer" in body or "CodeBERT" in body

    def test_approve_verdict_body(self):
        body = _build_review_body(EMPTY_REVIEW, [])
        assert "APPROVE" in body or "Approve" in body


class TestInlineBodyFormatter:
    def test_inline_body_contains_severity(self):
        c = {"severity": "CRITICAL", "category": "SECURITY",
             "comment": "SQL injection", "suggestion": "fix it"}
        body = _build_inline_body(c)
        assert "CRITICAL" in body
        assert "SECURITY" in body
        assert "SQL injection" in body

    def test_inline_body_contains_suggestion(self):
        c = {"severity": "HIGH", "category": "SECURITY",
             "comment": "Issue", "suggestion": "cursor.execute(q, (v,))"}
        body = _build_inline_body(c)
        assert "cursor.execute" in body

    def test_inline_body_no_suggestion_when_empty(self):
        c = {"severity": "LOW", "category": "STYLE", "comment": "Note", "suggestion": ""}
        body = _build_inline_body(c)
        assert "Suggested fix" not in body

    def test_inline_body_severity_emoji(self):
        for severity, emoji in SEVERITY_EMOJI.items():
            c = {"severity": severity, "category": "SECURITY", "comment": "x", "suggestion": ""}
            body = _build_inline_body(c)
            assert emoji in body


# ─── Integration Tests (mocked HTTP) ─────────────────────────────────────────

def _make_mock_response(status=200, json_data=None, raise_http=False):
    """Helper to make a mock requests.Response."""
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = json_data or {}
    mock.text = json.dumps(json_data or {})
    if raise_http:
        from requests.exceptions import HTTPError
        mock.raise_for_status.side_effect = HTTPError(response=mock)
    else:
        mock.raise_for_status.return_value = None
    return mock


class TestPostReviewToGithub:
    def test_missing_token_returns_error(self):
        import tools.github_poster as poster_module
        original = poster_module.GITHUB_TOKEN
        try:
            poster_module.GITHUB_TOKEN = ""
            result = post_review_to_github(
                "https://github.com/o/r/pull/1",
                SAMPLE_REVIEW,
                github_token="",
            )
        finally:
            poster_module.GITHUB_TOKEN = original
        assert result.success is False
        assert "GITHUB_TOKEN" in result.error

    def test_invalid_url_returns_error(self):
        result = post_review_to_github(
            "https://not-github.com/something",
            SAMPLE_REVIEW,
            github_token="fake_token",
        )
        assert result.success is False
        assert result.error is not None

    @patch("tools.github_poster.requests.get")
    @patch("tools.github_poster.requests.post")
    def test_successful_post_all_inline(self, mock_post, mock_get):
        """All comments land inline when their lines are in the diff."""
        mock_get.side_effect = [
            # First call: get head SHA
            _make_mock_response(json_data={"head": {"sha": "abc123"}}),
            # Second call: get diff files
            _make_mock_response(json_data=[
                {"filename": "db.py",   "patch": "@@ -1,3 +1,10 @@\n+line1\n+line2\n+line3\n+line4\n+line5\n"},
                {"filename": "auth.py", "patch": "@@ -1,5 +1,25 @@\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n+l\n"},
            ]),
        ]
        mock_post.return_value = _make_mock_response(json_data={
            "id": 12345,
            "html_url": "https://github.com/o/r/pull/1#pullrequestreview-12345",
        })

        result = post_review_to_github(
            "https://github.com/o/r/pull/1",
            SAMPLE_REVIEW,
            github_token="fake_token",
        )

        assert result.success is True
        assert result.review_id == 12345
        assert "github.com" in result.review_url

    @patch("tools.github_poster.requests.get")
    @patch("tools.github_poster.requests.post")
    def test_fallback_when_file_not_in_diff(self, mock_post, mock_get):
        """Comments on files NOT in the diff at all fall back to review body."""
        mock_get.side_effect = [
            _make_mock_response(json_data={"head": {"sha": "abc123"}}),
            # Only other_file.py is in the diff — db.py and auth.py are absent
            _make_mock_response(json_data=[
                {"filename": "other_file.py", "patch": "@@ -1,1 +1,5 @@\n line1\n+line2\n+line3\n+line4\n+line5"},
            ]),
        ]
        mock_post.return_value = _make_mock_response(json_data={
            "id": 999,
            "html_url": "https://github.com/o/r/pull/1#review-999",
        })

        result = post_review_to_github(
            "https://github.com/o/r/pull/1",
            SAMPLE_REVIEW,
            github_token="fake_token",
        )

        assert result.success is True
        # db.py and auth.py not in diff at all → both fall back
        assert result.inline_fallback == 2
        assert result.inline_posted == 0

        # Verify fallback comments appear in the review body
        posted_body = mock_post.call_args[1]["json"]["body"]
        assert "Additional Findings" in posted_body

    @patch("tools.github_poster.requests.get")
    @patch("tools.github_poster.requests.post")
    def test_snaps_to_nearest_diff_line(self, mock_post, mock_get):
        """Comments whose exact line isn't in the diff get snapped to nearest changed line."""
        # db.py lines 1-5 in diff, auth.py lines 1-20 in diff
        # agent comments on line 5 (db.py) and line 20 (auth.py) — exact match
        # but also tests a comment on line 99 (db.py) → should snap to line 5
        review_with_far_line = dict(SAMPLE_REVIEW)
        review_with_far_line["inline_comments"] = [
            {"file": "db.py", "line": 99, "severity": "HIGH", "category": "SECURITY",
             "comment": "Far line", "suggestion": ""},
        ]
        mock_get.side_effect = [
            _make_mock_response(json_data={"head": {"sha": "abc123"}}),
            _make_mock_response(json_data=[
                {"filename": "db.py", "patch": "@@ -1,3 +1,5 @@\n line1\n+line2\n+line3\n+line4\n+line5"},
            ]),
        ]
        mock_post.return_value = _make_mock_response(json_data={
            "id": 998, "html_url": "https://github.com/o/r/pull/1#review-998"
        })

        result = post_review_to_github(
            "https://github.com/o/r/pull/1",
            review_with_far_line,
            github_token="fake_token",
        )

        assert result.success is True
        assert result.inline_posted == 1   # snapped inline
        assert result.inline_fallback == 0 # not fallen back
        # Check the snapped line number is one of the valid diff lines (1-5)
        posted_comments = mock_post.call_args[1]["json"]["comments"]
        assert len(posted_comments) == 1
        assert posted_comments[0]["line"] in range(1, 6)  # snapped to nearest

    @patch("tools.github_poster.requests.get")
    @patch("tools.github_poster.requests.post")
    def test_approve_verdict_posts_approve_event(self, mock_post, mock_get):
        mock_get.side_effect = [
            _make_mock_response(json_data={"head": {"sha": "abc123"}}),
            _make_mock_response(json_data=[]),
        ]
        mock_post.return_value = _make_mock_response(json_data={
            "id": 1, "html_url": "https://github.com/o/r/pull/1#review-1"
        })

        post_review_to_github("https://github.com/o/r/pull/1", EMPTY_REVIEW,
                              github_token="fake_token")

        posted_event = mock_post.call_args[1]["json"]["event"]
        assert posted_event == "APPROVE"

    @patch("tools.github_poster.requests.get")
    @patch("tools.github_poster.requests.post")
    def test_request_changes_verdict_posts_correct_event(self, mock_post, mock_get):
        mock_get.side_effect = [
            _make_mock_response(json_data={"head": {"sha": "abc123"}}),
            _make_mock_response(json_data=[]),
        ]
        mock_post.return_value = _make_mock_response(json_data={
            "id": 2, "html_url": "https://github.com/o/r/pull/1#review-2"
        })

        post_review_to_github("https://github.com/o/r/pull/1", SAMPLE_REVIEW,
                              github_token="fake_token")

        posted_event = mock_post.call_args[1]["json"]["event"]
        assert posted_event == "REQUEST_CHANGES"

    @patch("tools.github_poster.requests.get")
    @patch("tools.github_poster.requests.post")
    def test_github_api_error_returns_failure(self, mock_post, mock_get):
        """GitHub returning 422 should produce a PostResult with success=False."""
        mock_get.return_value = _make_mock_response(json_data={"head": {"sha": "abc123"}})
        mock_post.return_value = _make_mock_response(
            status=422,
            json_data={"message": "Validation Failed"},
            raise_http=True,
        )

        result = post_review_to_github("https://github.com/o/r/pull/1", SAMPLE_REVIEW,
                                       github_token="fake_token")

        assert result.success is False
        assert result.error is not None

    @patch("tools.github_poster.requests.get")
    @patch("tools.github_poster.requests.post")
    def test_commit_sha_included_in_review(self, mock_post, mock_get):
        """The review must include the head commit SHA."""
        mock_get.side_effect = [
            _make_mock_response(json_data={"head": {"sha": "deadbeef"}}),
            _make_mock_response(json_data=[]),
        ]
        mock_post.return_value = _make_mock_response(json_data={
            "id": 3, "html_url": "https://github.com/o/r/pull/1#review-3"
        })

        post_review_to_github("https://github.com/o/r/pull/1", EMPTY_REVIEW,
                              github_token="fake_token")

        payload = mock_post.call_args[1]["json"]
        assert payload["commit_id"] == "deadbeef"

    @patch("tools.github_poster.requests.get")
    @patch("tools.github_poster.requests.post")
    def test_post_result_counts_correct(self, mock_post, mock_get):
        """inline_posted + inline_fallback = total inline comments."""
        mock_get.side_effect = [
            _make_mock_response(json_data={"head": {"sha": "abc123"}}),
            _make_mock_response(json_data=[
                {"filename": "db.py", "patch": "@@ -1,3 +1,10 @@\n+l\n+l\n+l\n+l\n+l\n"},
            ]),
        ]
        mock_post.return_value = _make_mock_response(json_data={
            "id": 4, "html_url": "https://github.com/o/r/pull/1#review-4"
        })

        result = post_review_to_github("https://github.com/o/r/pull/1", SAMPLE_REVIEW,
                                       github_token="fake_token")

        total = result.inline_posted + result.inline_fallback
        assert total == len(SAMPLE_REVIEW["inline_comments"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
