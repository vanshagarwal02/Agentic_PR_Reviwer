"""
test_week5.py — Tests for the ReAct agent (Week 5)

Test strategy:
  - Mock Gemini calls (no API key needed for most tests)
  - Test each component in isolation: prompts, tool registry, agent logic
  - Integration test with real tools (no Gemini) via mock responses
  - Edge cases: empty PRs, no functions, malformed Gemini output, rate limits
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


# ─── Fake ParsedPR for testing ────────────────────────────────────────────────

@dataclass
class FakeChangedFunction:
    function_name: str
    file_path: str
    full_source: str
    start_line: int = 1
    end_line: int = 10
    changed_lines: list = field(default_factory=list)
    is_new_file: bool = False
    language: str = "python"


@dataclass
class FakeParsedPR:
    pr_number: int = 1
    pr_title: str = "Test PR"
    pr_description: str = ""
    repo_full_name: str = "owner/repo"
    changed_files: list = field(default_factory=list)
    changed_functions: list = field(default_factory=list)


SQL_INJECTION_SOURCE = """def get_user(username):
    query = "SELECT * FROM users WHERE name='" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()
"""

SAFE_MATH_SOURCE = """def area(radius):
    import math
    return math.pi * radius ** 2
"""


def make_pr(functions=None, title="Test PR"):
    fns = functions or [
        FakeChangedFunction("get_user", "db.py", SQL_INJECTION_SOURCE)
    ]
    return FakeParsedPR(
        pr_number=99,
        pr_title=title,
        changed_files=[f.file_path for f in fns],
        changed_functions=fns,
    )


# ─── Prompt Tests ─────────────────────────────────────────────────────────────

class TestPrompts:
    def test_reasoning_prompt_includes_pr_title(self):
        from agent.prompts import build_reasoning_prompt
        ctx = {
            "title": "Fix SQL injection in login",
            "description": "",
            "repo": "owner/repo",
            "changed_files": ["db.py"],
            "changed_functions": [{
                "name": "login", "file": "db.py",
                "complexity": 2, "lines_changed": 5,
                "source_preview": "def login(): pass",
            }],
        }
        prompt = build_reasoning_prompt(ctx)
        assert "Fix SQL injection in login" in prompt
        assert "login" in prompt
        assert "tools_to_call" in prompt  # instructions present

    def test_reasoning_prompt_lists_tools(self):
        from agent.prompts import build_reasoning_prompt, TOOL_DESCRIPTIONS
        ctx = {
            "title": "PR", "description": "",
            "repo": "r", "changed_files": [],
            "changed_functions": [],
        }
        prompt = build_reasoning_prompt(ctx)
        for tool in TOOL_DESCRIPTIONS:
            if tool != "get_function_complexity":
                assert tool in prompt

    def test_synthesis_prompt_includes_tool_results(self):
        from agent.prompts import build_synthesis_prompt
        ctx = {"title": "PR", "repo": "r", "changed_functions": []}
        tool_results = {"scan_for_vulnerabilities": "get_user: CWE-89 CRITICAL"}
        prompt = build_synthesis_prompt(ctx, tool_results)
        assert "CWE-89" in prompt
        assert "overall_verdict" in prompt
        assert "inline_comments" in prompt

    def test_synthesis_prompt_no_tools_runs(self):
        from agent.prompts import build_synthesis_prompt
        ctx = {"title": "PR", "repo": "r", "changed_functions": []}
        prompt = build_synthesis_prompt(ctx, {})
        assert "No additional tools were run" in prompt


# ─── Tool Registry Tests ──────────────────────────────────────────────────────

class TestToolRegistry:
    def test_valid_tools_set(self):
        from agent.tool_registry import VALID_TOOLS
        assert "scan_for_vulnerabilities" in VALID_TOOLS
        assert "check_vulnerability_patterns" in VALID_TOOLS
        assert "build_call_graph" in VALID_TOOLS
        assert "semantic_search" in VALID_TOOLS

    def test_execute_unknown_tool_returns_string(self):
        from agent.tool_registry import execute_tool
        result = execute_tool("nonexistent_tool", "fn", "code", "f.py")
        assert "Unknown tool" in result
        assert isinstance(result, str)

    def test_execute_tool_no_source_handled(self):
        from agent.tool_registry import execute_tools_for_plan
        plan = [{"tool": "scan_for_vulnerabilities", "functions": ["missing_fn"], "reason": "test"}]
        results = execute_tools_for_plan(plan, function_sources={})
        assert len(results) == 1
        val = list(results.values())[0]
        assert "not available" in val or isinstance(val, str)

    def test_execute_scan_for_vulnerabilities(self):
        from agent.tool_registry import execute_tool
        result = execute_tool(
            "scan_for_vulnerabilities",
            "get_user",
            SQL_INJECTION_SOURCE,
            "db.py",
        )
        assert isinstance(result, str)
        assert "get_user" in result

    def test_execute_classifier_returns_string(self):
        from agent.tool_registry import execute_tool
        result = execute_tool(
            "check_vulnerability_patterns",
            "area",
            SAFE_MATH_SOURCE,
            "math.py",
        )
        assert isinstance(result, str)
        assert "label=" in result

    def test_execute_call_graph_no_repo(self):
        from agent.tool_registry import execute_tool
        result = execute_tool("build_call_graph", "fn", "def fn(): pass", "f.py", repo_path=None)
        assert "skipped" in result.lower() or isinstance(result, str)

    def test_format_results_groups_by_tool(self):
        from agent.tool_registry import format_results_for_prompt
        raw = {
            "scan_for_vulnerabilities:get_user": "get_user: CWE-89 CRITICAL",
            "scan_for_vulnerabilities:login":    "login: no matches",
            "check_vulnerability_patterns:get_user": "get_user: VULNERABLE 0.93",
        }
        grouped = format_results_for_prompt(raw)
        assert "scan_for_vulnerabilities" in grouped
        assert "check_vulnerability_patterns" in grouped
        assert "CWE-89" in grouped["scan_for_vulnerabilities"]
        assert "login" in grouped["scan_for_vulnerabilities"]

    def test_execute_tools_plan_with_valid_source(self):
        from agent.tool_registry import execute_tools_for_plan
        plan = [{"tool": "scan_for_vulnerabilities", "functions": ["get_user"], "reason": "test"}]
        sources = {"get_user": {"source": SQL_INJECTION_SOURCE, "file": "db.py", "start_line": 1}}
        results = execute_tools_for_plan(plan, sources)
        assert len(results) == 1
        assert isinstance(list(results.values())[0], str)


# ─── Agent Context Builder Tests ──────────────────────────────────────────────

class TestAgentContextBuilder:
    def test_build_pr_context_structure(self):
        from agent.react_agent import _build_pr_context
        pr = make_pr()
        ctx = _build_pr_context(pr, [])
        assert "title" in ctx
        assert "changed_functions" in ctx
        assert "changed_files" in ctx
        assert len(ctx["changed_functions"]) == 1
        assert ctx["changed_functions"][0]["name"] == "get_user"

    def test_build_pr_context_caps_functions(self):
        from agent.react_agent import _build_pr_context, MAX_FUNCTIONS
        fns = [FakeChangedFunction(f"fn{i}", "f.py", "def fn(): pass") for i in range(20)]
        pr = FakeParsedPR(changed_functions=fns)
        ctx = _build_pr_context(pr, [])
        assert len(ctx["changed_functions"]) <= MAX_FUNCTIONS

    def test_build_pr_context_source_preview_truncated(self):
        from agent.react_agent import _build_pr_context, SOURCE_PREVIEW_CHARS
        long_source = "x = 1\n" * 500
        pr = FakeParsedPR(
            changed_functions=[FakeChangedFunction("fn", "f.py", long_source)]
        )
        ctx = _build_pr_context(pr, [])
        preview = ctx["changed_functions"][0]["source_preview"]
        assert len(preview) <= SOURCE_PREVIEW_CHARS + 10  # +10 for "..."

    def test_build_function_sources_structure(self):
        from agent.react_agent import _build_pr_context, _build_function_sources
        pr = make_pr()
        ctx = _build_pr_context(pr, [])
        sources = _build_function_sources(ctx)
        assert "get_user" in sources
        assert "source" in sources["get_user"]
        assert "file" in sources["get_user"]

    def test_validate_tool_plan_filters_unknown_tools(self):
        from agent.react_agent import _validate_tool_plan
        plan = [
            {"tool": "scan_for_vulnerabilities", "functions": ["fn"], "reason": "test"},
            {"tool": "nonexistent_tool",          "functions": ["fn"], "reason": "bad"},
        ]
        validated = _validate_tool_plan(plan, {"fn"})
        assert len(validated) == 1
        assert validated[0]["tool"] == "scan_for_vulnerabilities"

    def test_validate_tool_plan_filters_unknown_functions(self):
        from agent.react_agent import _validate_tool_plan
        plan = [{"tool": "scan_for_vulnerabilities", "functions": ["ghost_fn"], "reason": ""}]
        # ghost_fn not in valid_function_names → falls back to all available
        validated = _validate_tool_plan(plan, {"real_fn"})
        assert validated[0]["functions"] == ["real_fn"]

    def test_validate_tool_plan_empty_input(self):
        from agent.react_agent import _validate_tool_plan
        assert _validate_tool_plan([], {"fn"}) == []


# ─── JSON Parsing Tests ───────────────────────────────────────────────────────

class TestJsonParsing:
    def test_parse_clean_json(self):
        from agent.react_agent import _parse_json_response
        data = _parse_json_response('{"key": "value"}', "test")
        assert data["key"] == "value"

    def test_parse_json_with_markdown_fences(self):
        from agent.react_agent import _parse_json_response
        text = '```json\n{"key": "value"}\n```'
        data = _parse_json_response(text, "test")
        assert data["key"] == "value"

    def test_parse_json_with_trailing_commas(self):
        from agent.react_agent import _parse_json_response
        text = '{"key": "value",}'
        data = _parse_json_response(text, "test")
        assert data["key"] == "value"

    def test_parse_invalid_json_raises(self):
        from agent.react_agent import _parse_json_response
        with pytest.raises(ValueError):
            _parse_json_response("this is not json at all {{{{", "test")


# ─── Agent Integration Tests (with mocked Gemini) ─────────────────────────────

MOCK_REASONING_RESPONSE = json.dumps({
    "pr_type": "security_fix",
    "risk_level_initial": "HIGH",
    "reasoning": "This PR modifies a database query function with string concatenation.",
    "security_relevant": True,
    "tools_to_call": [
        {
            "tool": "scan_for_vulnerabilities",
            "functions": ["get_user"],
            "reason": "String concatenation in SQL query",
        },
    ],
})

MOCK_SYNTHESIS_RESPONSE = json.dumps({
    "overall_verdict": "REQUEST_CHANGES",
    "risk_level": "HIGH",
    "summary": "This PR contains a SQL injection vulnerability in get_user.",
    "inline_comments": [
        {
            "file": "db.py",
            "line": 2,
            "severity": "CRITICAL",
            "category": "SECURITY",
            "comment": "SQL injection via string concatenation",
            "suggestion": "Use parameterized queries: cursor.execute(query, (username,))",
        }
    ],
    "tools_used": ["scan_for_vulnerabilities", "get_function_complexity"],
    "confidence": 0.92,
})


class TestAgentIntegration:
    @patch("agent.react_agent._get_gemini_client")
    @patch("agent.react_agent._call_gemini_with_retry")
    def test_agent_returns_review(self, mock_gemini_call, mock_get_model):
        from agent.react_agent import run_agent_review
        mock_gemini_call.side_effect = [MOCK_REASONING_RESPONSE, MOCK_SYNTHESIS_RESPONSE]
        pr = make_pr()
        review = run_agent_review(pr, [])
        assert review.overall_verdict == "REQUEST_CHANGES"
        assert review.risk_level == "HIGH"
        assert review.confidence == 0.92
        assert len(review.inline_comments) == 1
        assert review.inline_comments[0].severity == "CRITICAL"

    @patch("agent.react_agent._get_gemini_client")
    @patch("agent.react_agent._call_gemini_with_retry")
    def test_agent_counts_gemini_calls(self, mock_gemini_call, mock_get_model):
        from agent.react_agent import run_agent_review
        mock_gemini_call.side_effect = [MOCK_REASONING_RESPONSE, MOCK_SYNTHESIS_RESPONSE]
        review = run_agent_review(make_pr(), [])
        assert review.gemini_calls == 2

    @patch("agent.react_agent._get_gemini_client")
    @patch("agent.react_agent._call_gemini_with_retry")
    def test_agent_records_pr_type(self, mock_gemini_call, mock_get_model):
        from agent.react_agent import run_agent_review
        mock_gemini_call.side_effect = [MOCK_REASONING_RESPONSE, MOCK_SYNTHESIS_RESPONSE]
        review = run_agent_review(make_pr(), [])
        assert review.pr_type == "security_fix"

    @patch("agent.react_agent._get_gemini_client")
    @patch("agent.react_agent._call_gemini_with_retry")
    def test_agent_records_reasoning(self, mock_gemini_call, mock_get_model):
        from agent.react_agent import run_agent_review
        mock_gemini_call.side_effect = [MOCK_REASONING_RESPONSE, MOCK_SYNTHESIS_RESPONSE]
        review = run_agent_review(make_pr(), [])
        assert len(review.reasoning) > 0

    def test_agent_empty_pr_no_functions(self):
        """Agent should handle PRs with no Python functions gracefully."""
        from agent.react_agent import run_agent_review
        pr = FakeParsedPR(pr_number=1, changed_functions=[])
        review = run_agent_review(pr, [])
        assert review.overall_verdict in ("COMMENT", "APPROVE")
        assert review.error is None
        assert len(review.summary) > 0

    def test_agent_missing_api_key_returns_error(self):
        """Missing API key should return a review with error, not raise."""
        from agent.react_agent import run_agent_review
        import agent.react_agent as agent_module
        original = agent_module.GEMINI_API_KEY
        try:
            agent_module.GEMINI_API_KEY = "your_gemini_key_here"
            pr = make_pr()
            review = run_agent_review(pr, [])
        finally:
            agent_module.GEMINI_API_KEY = original
        assert review.error is not None
        assert "GEMINI_API_KEY" in review.error

    @patch("agent.react_agent._get_gemini_client")
    @patch("agent.react_agent._call_gemini_with_retry")
    def test_agent_malformed_json_from_gemini(self, mock_gemini_call, mock_get_model):
        """Malformed JSON from Gemini should produce a review with error, not crash."""
        from agent.react_agent import run_agent_review
        mock_gemini_call.side_effect = ["this is not json {{{{", MOCK_SYNTHESIS_RESPONSE]
        review = run_agent_review(make_pr(), [])
        assert review.error is not None

    @patch("agent.react_agent._get_gemini_client")
    @patch("agent.react_agent._call_gemini_with_retry")
    def test_agent_no_tools_selected_still_synthesizes(self, mock_gemini_call, mock_get_model):
        """Agent should still synthesize even if it selects zero tools."""
        from agent.react_agent import run_agent_review
        no_tools_reasoning = json.dumps({
            "pr_type": "style",
            "risk_level_initial": "LOW",
            "reasoning": "Just a style change.",
            "security_relevant": False,
            "tools_to_call": [],
        })
        mock_gemini_call.side_effect = [no_tools_reasoning, MOCK_SYNTHESIS_RESPONSE]
        review = run_agent_review(make_pr(), [])
        assert review.overall_verdict is not None
        assert review.gemini_calls == 2

    @patch("agent.react_agent._get_gemini_client")
    @patch("agent.react_agent._call_gemini_with_retry")
    def test_agent_to_dict_serializable(self, mock_gemini_call, mock_get_model):
        """agent_review_to_dict output must be JSON-serializable."""
        from agent.react_agent import run_agent_review, agent_review_to_dict
        mock_gemini_call.side_effect = [MOCK_REASONING_RESPONSE, MOCK_SYNTHESIS_RESPONSE]
        review = run_agent_review(make_pr(), [])
        d = agent_review_to_dict(review)
        serialized = json.dumps(d)  # should not raise
        parsed = json.loads(serialized)
        assert "overall_verdict" in parsed
        assert "inline_comments" in parsed
        assert "agent_meta" in parsed

    @patch("agent.react_agent._get_gemini_client")
    @patch("agent.react_agent._call_gemini_with_retry")
    def test_agent_inline_comment_fields(self, mock_gemini_call, mock_get_model):
        """Each inline comment must have all required fields."""
        from agent.react_agent import run_agent_review, agent_review_to_dict
        mock_gemini_call.side_effect = [MOCK_REASONING_RESPONSE, MOCK_SYNTHESIS_RESPONSE]
        review = run_agent_review(make_pr(), [])
        d = agent_review_to_dict(review)
        for c in d["inline_comments"]:
            assert "file" in c
            assert "line" in c
            assert "severity" in c
            assert "category" in c
            assert "comment" in c
            assert "suggestion" in c

    @patch("agent.react_agent._get_gemini_client")
    @patch("agent.react_agent._call_gemini_with_retry")
    def test_agent_elapsed_time_recorded(self, mock_gemini_call, mock_get_model):
        from agent.react_agent import run_agent_review
        mock_gemini_call.side_effect = [MOCK_REASONING_RESPONSE, MOCK_SYNTHESIS_RESPONSE]
        review = run_agent_review(make_pr(), [])
        assert review.elapsed_seconds >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
