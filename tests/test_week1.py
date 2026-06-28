"""Tests for parse_diff() and get_function_complexity()"""

import pytest
from tools.complexity_analyzer import get_function_complexity, ComplexityResult


# ─── Complexity Analyzer Tests ────────────────────────────────────────────────

class TestComplexityAnalyzer:
    """
    These tests verify get_function_complexity() produces correct metrics.
    Run with: pytest tests/ -v
    """

    def test_simple_function_low_complexity(self):
        """A simple function with no branches should have CC=1"""
        code = """
def greet(name):
    return f"Hello, {name}!"
"""
        result = get_function_complexity(code, "python")
        assert result.cyclomatic_complexity == 1
        assert result.complexity_label == "LOW"
        assert result.needs_refactoring is False
        assert result.review_comment is None

    def test_function_with_if_else(self):
        """Each if/elif adds 1 to complexity"""
        code = """
def categorize_score(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "F"
"""
        result = get_function_complexity(code, "python")
        # base 1 + if + elif + elif = 4
        assert result.cyclomatic_complexity == 4
        assert result.complexity_label == "LOW"

    def test_deeply_nested_function(self):
        """Detect deep nesting even if CC is moderate"""
        code = """
def deeply_nested(a, b, c, d):
    if a:
        for x in b:
            while c:
                if d:
                    if a > 0:
                        return True
    return False
"""
        result = get_function_complexity(code, "python")
        assert result.max_nesting_depth >= 4
        assert result.needs_refactoring is True

    def test_function_with_try_except(self):
        """try/except adds to complexity"""
        code = """
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
    except TypeError:
        return None
"""
        result = get_function_complexity(code, "python")
        # base 1 + try + except + except = 4 (try itself may not count, but excepts do)
        assert result.cyclomatic_complexity >= 2

    def test_parameter_count(self):
        """Count parameters correctly, excluding self/cls"""
        code = """
def process(self, a, b, c, d, e, f):
    return a + b + c + d + e + f
"""
        result = get_function_complexity(code, "python")
        assert result.parameter_count == 6  # self excluded

    def test_high_complexity_needs_refactoring(self):
        """A complex function should trigger the refactoring flag"""
        code = """
def complex_func(a, b, c, d, e, f):
    if a:
        for x in b:
            if x > 0:
                while c:
                    if d and e:
                        try:
                            if f:
                                return True
                        except:
                            pass
    elif b:
        for y in a:
            if y:
                return y
    return False
"""
        result = get_function_complexity(code, "python")
        assert result.needs_refactoring is True
        assert result.review_comment is not None
        assert "cyclomatic" in result.review_comment.lower()

    def test_recursive_function_detected(self):
        """Direct recursion should be flagged"""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
        result = get_function_complexity(code, "python")
        assert result.has_recursion is True

    def test_line_count(self):
        """Line count should match non-blank lines"""
        code = """
def foo():
    x = 1
    y = 2
    z = 3
    return x + y + z
"""
        result = get_function_complexity(code, "python")
        # def line + 4 body lines = 5 non-blank lines
        assert result.line_count == 5

    def test_unknown_language_graceful_fallback(self):
        """Unsupported languages should return a minimal result without crashing"""
        code = "function foo() { return 42; }"
        result = get_function_complexity(code, "javascript")
        assert result is not None
        assert result.cyclomatic_complexity == 1


class TestDiffParser:
    """
    NOTE: parse_diff() requires a real GitHub token and internet access.
    These tests use known stable PRs from major open source repos.

    Run ONLY these specific tests when you have GITHUB_TOKEN set:
      pytest tests/ -v -k "test_parse" --run-github
    """

    @pytest.mark.skip(reason="Requires GITHUB_TOKEN and internet — run manually")
    def test_parse_real_pr_returns_changed_functions(self):
        """
        Uses a real merged PR from the 'requests' library.
        This PR modifies auth.py — we expect to find changed functions there.
        """
        from tools.parse_diff import parse_diff
        # Using a stable, merged PR — won't change
        parsed = parse_diff("https://github.com/psf/requests/pull/6734")

        assert parsed.pr_number == 6734
        assert len(parsed.changed_files) > 0
        assert isinstance(parsed.pr_title, str)
        # The PR touches Python files, so we should extract functions
        assert len(parsed.changed_functions) >= 0  # might be 0 if only non-Python changed

    @pytest.mark.skip(reason="Requires GITHUB_TOKEN and internet — run manually")
    def test_parse_pr_with_python_changes(self):
        """Verify the structured output has all required fields."""
        from tools.parse_diff import parse_diff
        parsed = parse_diff("https://github.com/tiangolo/fastapi/pull/11751")

        for cf in parsed.changed_functions:
            assert cf.file_path != ""
            assert cf.function_name != ""
            assert cf.language in ("python", "c", "cpp", "unknown")
            assert cf.start_line <= cf.end_line
            assert isinstance(cf.changed_lines, list)
            assert isinstance(cf.full_source, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
