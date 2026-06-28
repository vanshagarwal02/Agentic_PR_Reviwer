"""Tests for call_graph.py and semantic_search.py (Week 2 tools)"""

import pytest
from tools.call_graph import (
    build_call_graph,
    _extract_functions_and_calls,
    _build_graph,
    _compute_impact,
    FunctionNode,
)
from tools.semantic_search import (
    semantic_search,
    _tokenize_code,
    _compute_tf,
    _compute_idf,
    _tfidf_vector,
    _cosine_similarity,
    _scan_repo_for_functions,
)


# ─── Call Graph Tests ─────────────────────────────────────────────────────────

class TestCallGraphExtraction:
    """Test tree-sitter-based function and call extraction."""

    def test_extracts_function_names(self):
        """Basic function definitions should be found."""
        source = """
def alpha():
    pass

def beta():
    pass

def gamma():
    pass
"""
        fns = _extract_functions_and_calls(source, "alpha.py")
        names = [f.name for f in fns]
        assert "alpha" in names
        assert "beta" in names
        assert "gamma" in names

    def test_detects_direct_calls(self):
        """When alpha() calls beta(), the call should be recorded."""
        source = """
def beta():
    return 42

def alpha():
    result = beta()
    return result
"""
        fns = _extract_functions_and_calls(source, "test.py")
        alpha_node = next(f for f in fns if f.name == "alpha")
        assert "beta" in alpha_node.calls

    def test_detects_method_calls(self):
        """obj.method() calls should record the method name."""
        source = """
def do_something(self):
    self.helper()
    self.logger.info("done")
"""
        fns = _extract_functions_and_calls(source, "test.py")
        do_fn = fns[0]
        # Should detect "helper" and "info" as called names
        assert "helper" in do_fn.calls or "info" in do_fn.calls

    def test_nested_functions_found(self):
        """Inner functions should also be extracted."""
        source = """
def outer():
    def inner():
        pass
    inner()
"""
        fns = _extract_functions_and_calls(source, "test.py")
        names = [f.name for f in fns]
        assert "outer" in names
        assert "inner" in names

    def test_decorated_function_found(self):
        """Functions with decorators should still be extracted."""
        source = """
@property
def my_prop(self):
    return self._value

@staticmethod
def create():
    return MyClass()
"""
        fns = _extract_functions_and_calls(source, "test.py")
        names = [f.name for f in fns]
        assert "my_prop" in names
        assert "create" in names

    def test_line_numbers_correct(self):
        """Start and end line numbers should be 1-indexed and correct."""
        source = "def foo():\n    return 1\n"
        fns = _extract_functions_and_calls(source, "test.py")
        assert len(fns) == 1
        assert fns[0].start_line == 1
        assert fns[0].end_line == 2


class TestCallGraphBuilding:
    """Test the networkx graph construction and impact scoring."""

    def _make_nodes(self):
        """Helper: create a small set of FunctionNodes for testing."""
        return [
            FunctionNode("a", "x.py", 1, 5, calls=["b", "c"]),
            FunctionNode("b", "x.py", 6, 10, calls=["c"]),
            FunctionNode("c", "x.py", 11, 15, calls=[]),
        ]

    def test_graph_has_correct_nodes(self):
        G = _build_graph(self._make_nodes())
        assert "a" in G.nodes
        assert "b" in G.nodes
        assert "c" in G.nodes

    def test_graph_has_correct_edges(self):
        G = _build_graph(self._make_nodes())
        assert G.has_edge("a", "b")
        assert G.has_edge("a", "c")
        assert G.has_edge("b", "c")
        assert not G.has_edge("c", "a")  # c calls nothing

    def test_impact_leaf_function(self):
        """A function that calls nothing is a leaf."""
        G = _build_graph(self._make_nodes())
        impact = _compute_impact("c", "x.py", G, max_callers=2)
        assert impact.is_leaf is True
        assert impact.fan_out == 0

    def test_impact_root_function(self):
        """A function that nothing calls is a root."""
        G = _build_graph(self._make_nodes())
        impact = _compute_impact("a", "x.py", G, max_callers=2)
        assert impact.is_root is True
        assert impact.fan_in == 0

    def test_impact_score_normalized(self):
        """Impact score should be between 0 and 1."""
        G = _build_graph(self._make_nodes())
        impact = _compute_impact("c", "x.py", G, max_callers=2)
        assert 0.0 <= impact.impact_score <= 1.0

    def test_high_impact_when_many_callers(self):
        """c is called by both a and b, so it has higher impact than a."""
        G = _build_graph(self._make_nodes())
        impact_c = _compute_impact("c", "x.py", G, max_callers=2)
        impact_a = _compute_impact("a", "x.py", G, max_callers=2)
        assert impact_c.impact_score > impact_a.impact_score

    def test_build_call_graph_on_own_repo(self):
        """
        Integration test: run on our own repo.
        Should find functions and build a non-empty graph.
        """
        import os
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        result = build_call_graph(repo_root, ["parse_diff", "get_function_complexity"])

        assert result.total_files_scanned > 0
        assert result.total_functions_found > 0
        assert result.graph is not None
        assert result.graph.number_of_nodes() > 0
        # Both query functions should have impact entries
        assert len(result.impacts) == 2
        names = [imp.function_name for imp in result.impacts]
        assert "parse_diff" in names
        assert "get_function_complexity" in names

    def test_unknown_function_returns_safe_impact(self):
        """Querying a function that doesn't exist should not crash."""
        import os
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        result = build_call_graph(repo_root, ["nonexistent_function_xyz"])
        assert len(result.impacts) == 1
        assert result.impacts[0].function_name == "nonexistent_function_xyz"
        assert result.impacts[0].impact_score == 0.0


# ─── Semantic Search Tests ────────────────────────────────────────────────────

class TestTokenization:
    """Test the code tokenization used by TF-IDF."""

    def test_basic_tokenization(self):
        tokens = _tokenize_code("def validate_token(token):\n    return True")
        assert "validate" in tokens or "token" in tokens

    def test_camelcase_split(self):
        tokens = _tokenize_code("validateToken = myFunction()")
        lowered = [t.lower() for t in tokens]
        # "validateToken" → ["validate", "token"] or similar
        assert "validate" in lowered or "token" in lowered

    def test_keywords_excluded(self):
        tokens = _tokenize_code("def return if else for while")
        # Python keywords should be stripped
        for kw in ["def", "return", "if", "else", "for", "while"]:
            assert kw not in tokens

    def test_short_tokens_excluded(self):
        tokens = _tokenize_code("a = b + c")
        # Single-character tokens should be excluded
        assert "a" not in tokens
        assert "b" not in tokens


class TestTFIDFMath:
    """Test the TF-IDF implementation correctness."""

    def test_tf_sums_less_than_one(self):
        """TF values should be normalized (each value ≤ 1)."""
        tokens = ["apple", "apple", "banana", "cherry"]
        tf = _compute_tf(tokens)
        for v in tf.values():
            assert 0.0 <= v <= 1.0

    def test_idf_rare_word_higher(self):
        """A word appearing in fewer docs should have higher IDF."""
        docs = [["common", "word", "here"], ["common", "stuff"], ["rare_xyz"]]
        idf = _compute_idf(docs)
        # "rare_xyz" appears in 1 doc, "common" in 2 — rare should have higher IDF
        assert idf.get("rare_xyz", 0) > idf.get("common", 0)

    def test_cosine_similarity_identical(self):
        """A vector compared to itself should give similarity 1.0."""
        vec = {"auth": 0.5, "token": 0.3, "validate": 0.2}
        sim = _cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal(self):
        """Two vectors with no shared words should give similarity 0.0."""
        vec_a = {"apple": 0.5, "banana": 0.5}
        vec_b = {"orange": 0.5, "grape": 0.5}
        sim = _cosine_similarity(vec_a, vec_b)
        assert sim == 0.0

    def test_cosine_similarity_range(self):
        """Similarity should always be between 0 and 1."""
        vec_a = {"auth": 0.5, "validate": 0.3, "token": 0.2}
        vec_b = {"auth": 0.4, "check": 0.4, "token": 0.2}
        sim = _cosine_similarity(vec_a, vec_b)
        assert 0.0 <= sim <= 1.0


class TestSemanticSearch:
    """Integration tests for the semantic_search() function."""

    def test_finds_similar_functions_in_own_repo(self):
        """Running on our own repo should return results without crashing."""
        import os
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        # Use parse_diff as query — it's a substantial function
        all_fns = _scan_repo_for_functions(repo_root)
        query_fn = next((f for f in all_fns if f.name == "parse_diff"), None)

        if query_fn is None:
            pytest.skip("parse_diff not found in scanned functions")

        result = semantic_search(
            query_source=query_fn.source,
            query_name=query_fn.name,
            query_file=query_fn.file_path,
            repo_path=repo_root,
            top_k=3,
        )

        assert result.mode_used == "tfidf"
        assert result.total_functions_searched > 0
        # All scores should be in [0, 1]
        for sf in result.similar_functions:
            assert 0.0 <= sf.similarity_score <= 1.0

    def test_no_self_match(self):
        """The query function itself should never appear in results."""
        import os
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        all_fns = _scan_repo_for_functions(repo_root)
        query_fn = next((f for f in all_fns if f.name == "parse_diff"), None)

        if query_fn is None:
            pytest.skip("parse_diff not found")

        result = semantic_search(
            query_source=query_fn.source,
            query_name=query_fn.name,
            query_file=query_fn.file_path,
            repo_path=repo_root,
        )
        for sf in result.similar_functions:
            assert not (sf.function_name == "parse_diff" and sf.file_path == query_fn.file_path)

    def test_results_sorted_by_score(self):
        """Results should be in descending similarity order."""
        import os
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        all_fns = _scan_repo_for_functions(repo_root)
        query_fn = next((f for f in all_fns if f.name == "parse_diff"), None)

        if query_fn is None:
            pytest.skip("parse_diff not found")

        result = semantic_search(
            query_source=query_fn.source,
            query_name=query_fn.name,
            query_file=query_fn.file_path,
            repo_path=repo_root,
        )

        scores = [sf.similarity_score for sf in result.similar_functions]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        """Should never return more than top_k results."""
        import os
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        all_fns = _scan_repo_for_functions(repo_root)
        query_fn = next((f for f in all_fns if f.name == "parse_diff"), None)

        if query_fn is None:
            pytest.skip("parse_diff not found")

        result = semantic_search(
            query_source=query_fn.source,
            query_name=query_fn.name,
            query_file=query_fn.file_path,
            repo_path=repo_root,
            top_k=2,
        )
        assert len(result.similar_functions) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
