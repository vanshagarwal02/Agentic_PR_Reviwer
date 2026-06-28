"""Tests for vulnerability_scanner.py (Week 3)"""

import pytest
from tools.vulnerability_scanner import (
    scan_for_vulnerabilities,
    ensure_index_exists,
    _embed,
    VULNERABILITY_PATTERNS,
    VulnerabilityMatch,
    VulnerabilityScanResult,
)


# Build/load the index once for all tests in this module
@pytest.fixture(scope="module", autouse=True)
def load_index():
    """Load the FAISS index once before any test in this module runs."""
    ensure_index_exists()


# ─── Embedding Tests ──────────────────────────────────────────────────────────

class TestEmbedding:
    """Test that CodeBERT produces valid embeddings."""

    def test_embedding_shape(self):
        """CodeBERT should return a 768-dim vector."""
        vec = _embed("def foo(): return 42")
        assert vec.shape == (768,)

    def test_embedding_normalized(self):
        """After L2 normalization, vector magnitude should be ~1.0."""
        import numpy as np
        vec = _embed("def authenticate(token): return token is not None")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_embedding_is_float32(self):
        """FAISS requires float32 — make sure dtype is correct."""
        import numpy as np
        vec = _embed("def foo(): pass")
        assert vec.dtype == np.float32

    def test_different_code_different_vectors(self):
        """Two structurally different functions should have different embeddings."""
        import numpy as np
        vec_a = _embed("def hash_password(pw): return hashlib.md5(pw).hexdigest()")
        vec_b = _embed("def open_file(path): return open(path).read()")
        # Cosine similarity should be < 1.0 (not identical)
        similarity = float(np.dot(vec_a, vec_b))
        assert similarity < 0.999

    def test_similar_code_similar_vectors(self):
        """Two very similar SQL injection patterns should have high cosine similarity."""
        import numpy as np
        sql1 = """def get_user(name):
    query = "SELECT * FROM users WHERE name = '" + name + "'"
    cursor.execute(query)"""
        sql2 = """def find_record(id):
    sql = "SELECT * FROM records WHERE id = '" + id + "'"
    db.execute(sql)"""
        vec_a = _embed(sql1)
        vec_b = _embed(sql2)
        similarity = float(np.dot(vec_a, vec_b))
        assert similarity > 0.95


# ─── Scanner Tests ────────────────────────────────────────────────────────────

class TestSQLInjectionDetection:
    """The scanner must detect SQL injection patterns."""

    def test_detects_string_concatenation_sqli(self):
        """Classic string concat SQL injection should be caught."""
        code = """def get_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()"""
        result = scan_for_vulnerabilities(code, "get_user", "app.py")
        assert result.scanned is True
        assert len(result.matches) > 0
        cwes = [m.cwe for m in result.matches]
        assert "CWE-89" in cwes

    def test_detects_fstring_sqli(self):
        """F-string SQL injection should be caught."""
        code = """def search(term):
    sql = f"SELECT * FROM products WHERE name LIKE '%{term}%'"
    db.execute(sql)
    return db.fetchall()"""
        result = scan_for_vulnerabilities(code, "search", "db.py")
        assert result.scanned is True
        cwes = [m.cwe for m in result.matches]
        assert "CWE-89" in cwes

    def test_severity_is_critical(self):
        """SQL injection should be flagged as CRITICAL severity."""
        code = """def get_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)"""
        result = scan_for_vulnerabilities(code, "get_user", "app.py")
        severities = [m.severity for m in result.matches]
        assert "CRITICAL" in severities

    def test_result_includes_fix(self):
        """Every match should include a non-empty fix suggestion."""
        code = """def get_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)"""
        result = scan_for_vulnerabilities(code, "get_user", "app.py")
        for match in result.matches:
            assert match.fix != ""

    def test_similarity_score_is_high(self):
        """Score should be ≥ 0.90 for near-identical patterns."""
        code = """def get_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()"""
        result = scan_for_vulnerabilities(code, "get_user", "app.py")
        sqli_matches = [m for m in result.matches if m.cwe == "CWE-89"]
        assert len(sqli_matches) > 0
        assert sqli_matches[0].similarity_score >= 0.90


class TestCommandInjectionDetection:
    """The scanner must detect command injection patterns."""

    def test_detects_os_system_injection(self):
        """os.system with user input should be flagged."""
        code = """def ping(host):
    os.system("ping -c 1 " + host)"""
        result = scan_for_vulnerabilities(code, "ping", "net.py")
        assert result.scanned is True
        cwes = [m.cwe for m in result.matches]
        assert "CWE-78" in cwes

    def test_detects_eval_injection(self):
        """eval() on user input should be flagged as command injection."""
        code = """def calculate(expression):
    result = eval(expression)
    return result"""
        result = scan_for_vulnerabilities(code, "calculate", "calc.py")
        assert result.scanned is True
        cwes = [m.cwe for m in result.matches]
        assert "CWE-78" in cwes


class TestWeakCryptographyDetection:
    """The scanner must detect weak hashing patterns."""

    def test_detects_md5_password_hashing(self):
        """MD5 for passwords is a known vulnerability (CWE-328)."""
        code = """def hash_password(password):
    import hashlib
    return hashlib.md5(password.encode()).hexdigest()"""
        result = scan_for_vulnerabilities(code, "hash_password", "auth.py")
        assert result.scanned is True
        cwes = [m.cwe for m in result.matches]
        assert "CWE-328" in cwes

    def test_detects_weak_random_token(self):
        """random.choice for security tokens should be flagged."""
        code = """def generate_token():
    import random, string
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(32))"""
        result = scan_for_vulnerabilities(code, "generate_token", "auth.py")
        assert result.scanned is True
        cwes = [m.cwe for m in result.matches]
        assert "CWE-327" in cwes


class TestHardcodedCredentials:
    """The scanner must detect hardcoded secrets."""

    def test_detects_hardcoded_password(self):
        """Hardcoded password string should trigger CWE-798."""
        code = """def connect():
    password = "admin123"
    return db.connect(host="localhost", user="admin", password=password)"""
        result = scan_for_vulnerabilities(code, "connect", "db.py")
        assert result.scanned is True
        cwes = [m.cwe for m in result.matches]
        assert "CWE-798" in cwes


class TestSafeCodeNotFlagged:
    """
    Safe code should score below a very high threshold.

    Note: CodeBERT embeds ALL short Python functions in a tight cluster
    (cosine similarity 0.99-1.0). The system works by distinguishing
    DEGREE of similarity — dangerous patterns score 0.999+, while safe
    code with different vocabulary scores 0.995 or lower.
    Using min_similarity=0.999 reliably separates safe from dangerous.
    """

    def test_parameterized_query_not_flagged(self):
        """Parameterized queries are safe and should score below 0.999."""
        code = """def get_user_safe(username):
    query = "SELECT * FROM users WHERE name = %s"
    cursor.execute(query, (username,))
    return cursor.fetchone()"""
        result = scan_for_vulnerabilities(
            code, "get_user_safe", "app.py", min_similarity=0.999
        )
        sqli = [m for m in result.matches if m.cwe == "CWE-89"]
        assert len(sqli) == 0

    def test_simple_math_not_flagged(self):
        """A math function with no security-relevant tokens should score < 0.999."""
        code = """def euclidean_distance(point_a, point_b):
    squared_diff = sum((a - b) ** 2 for a, b in zip(point_a, point_b))
    return squared_diff ** 0.5"""
        result = scan_for_vulnerabilities(
            code, "euclidean_distance", "math.py", min_similarity=0.999
        )
        assert len(result.matches) == 0


# ─── Result Structure Tests ───────────────────────────────────────────────────

class TestResultStructure:
    """Verify the shape of returned data."""

    def test_result_has_correct_fields(self):
        """VulnerabilityScanResult must have all expected fields."""
        code = "def foo(): return 1"
        result = scan_for_vulnerabilities(code, "foo", "test.py")
        assert isinstance(result, VulnerabilityScanResult)
        assert isinstance(result.function_name, str)
        assert isinstance(result.file_path, str)
        assert isinstance(result.scanned, bool)
        assert isinstance(result.matches, list)
        assert isinstance(result.functions_in_index, int)

    def test_match_has_all_fields(self):
        """Each VulnerabilityMatch must have all expected fields."""
        code = """def get_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)"""
        result = scan_for_vulnerabilities(code, "get_user", "app.py")
        assert len(result.matches) > 0
        m = result.matches[0]
        assert isinstance(m, VulnerabilityMatch)
        assert m.cwe.startswith("CWE-")
        assert m.severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        assert 0.0 <= m.similarity_score <= 1.0
        assert m.fix != ""
        assert m.description != ""

    def test_results_sorted_by_score(self):
        """Matches should be ordered by similarity score descending."""
        code = """def get_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)"""
        result = scan_for_vulnerabilities(code, "get_user", "app.py", top_k=3)
        scores = [m.similarity_score for m in result.matches]
        assert scores == sorted(scores, reverse=True)

    def test_index_has_all_patterns(self):
        """Index should contain all 21 defined patterns."""
        code = "def foo(): pass"
        result = scan_for_vulnerabilities(code, "foo", "test.py")
        assert result.functions_in_index == len(VULNERABILITY_PATTERNS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
