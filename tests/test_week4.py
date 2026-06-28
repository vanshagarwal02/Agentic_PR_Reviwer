"""Tests for vulnerability_classifier.py (Week 4)"""

import pytest
from tools.vulnerability_classifier import (
    check_vulnerability_patterns,
    classify_functions,
    _is_model_trained,
    _load_classifier,
    ClassifierPrediction,
    ClassificationResult,
)


# ─── Model Availability ───────────────────────────────────────────────────────

class TestModelAvailability:
    def test_model_trained_returns_bool(self):
        result = _is_model_trained()
        assert isinstance(result, bool)

    def test_load_classifier_returns_bool(self):
        result = _load_classifier()
        assert isinstance(result, bool)

    def test_model_or_fallback_works(self):
        """Either trained model or FAISS fallback — something must work."""
        code = "def add(a, b): return a + b"
        result = check_vulnerability_patterns(code, "add", "test.py")
        assert result.label in ("VULNERABLE", "SAFE", "UNKNOWN")


# ─── Output Structure ─────────────────────────────────────────────────────────

class TestOutputStructure:
    def test_returns_classifier_prediction(self):
        code = "def foo(): return 42"
        result = check_vulnerability_patterns(code, "foo", "test.py")
        assert isinstance(result, ClassifierPrediction)

    def test_all_fields_present(self):
        code = "def foo(): return 42"
        result = check_vulnerability_patterns(code, "foo", "test.py")
        assert isinstance(result.function_name, str)
        assert isinstance(result.file_path, str)
        assert result.label in ("VULNERABLE", "SAFE", "UNKNOWN")
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.vulnerable_probability <= 1.0
        assert isinstance(result.method, str)
        assert isinstance(result.fallback_used, bool)

    def test_function_name_preserved(self):
        code = "def my_func(): pass"
        result = check_vulnerability_patterns(code, "my_func", "mod.py")
        assert result.function_name == "my_func"

    def test_file_path_preserved(self):
        code = "def foo(): pass"
        result = check_vulnerability_patterns(code, "foo", "auth/login.py")
        assert result.file_path == "auth/login.py"

    def test_confidence_in_range(self):
        code = "def sq(n): return n*n"
        result = check_vulnerability_patterns(code, "sq", "math.py")
        assert 0.0 <= result.confidence <= 1.0

    def test_vuln_prob_in_range(self):
        code = "def sq(n): return n*n"
        result = check_vulnerability_patterns(code, "sq", "math.py")
        assert 0.0 <= result.vulnerable_probability <= 1.0


# ─── Batch Classification ─────────────────────────────────────────────────────

class TestClassifyFunctions:
    def test_returns_classification_result(self):
        fns = [{"source": "def foo(): return 1", "name": "foo", "file": "a.py"}]
        result = classify_functions(fns)
        assert isinstance(result, ClassificationResult)

    def test_returns_correct_count(self):
        fns = [
            {"source": "def foo(): return 1", "name": "foo", "file": "a.py"},
            {"source": "def bar(): return 2", "name": "bar", "file": "b.py"},
            {"source": "def baz(): return 3", "name": "baz", "file": "c.py"},
        ]
        result = classify_functions(fns)
        assert len(result.predictions) == 3

    def test_empty_batch_returns_empty(self):
        result = classify_functions([])
        assert isinstance(result, ClassificationResult)
        assert len(result.predictions) == 0

    def test_model_available_field(self):
        fns = [{"source": "def f(): pass", "name": "f", "file": "x.py"}]
        result = classify_functions(fns)
        assert isinstance(result.model_available, bool)

    def test_all_predictions_valid(self):
        fns = [
            {"source": "def a(): pass", "name": "a", "file": "a.py"},
            {"source": "def b(): return None", "name": "b", "file": "b.py"},
        ]
        result = classify_functions(fns)
        for pred in result.predictions:
            assert pred.label in ("VULNERABLE", "SAFE", "UNKNOWN")
            assert 0.0 <= pred.confidence <= 1.0


# ─── Classification Logic ─────────────────────────────────────────────────────

class TestClassificationLogic:
    """Test that the classifier distinguishes safe from vulnerable code."""

    def test_sql_injection_flagged(self):
        """
        Classic SQL injection should be caught.
        With the full Kaggle model: via binary classifier.
        With the mini model: falls back to FAISS scanner which always catches it.
        We test that AT LEAST ONE of the two layers flags it.
        """
        from tools.vulnerability_scanner import scan_for_vulnerabilities
        code = """def get_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()"""
        # FAISS scanner (Week 3) always catches this
        faiss_result = scan_for_vulnerabilities(code, "get_user", "db.py")
        faiss_flagged = bool(faiss_result.matches)
        # Classifier result (mini or full model)
        clf_result = check_vulnerability_patterns(code, "get_user", "db.py")
        # At least one layer must flag it
        assert faiss_flagged or clf_result.label == "VULNERABLE", (
            f"Neither FAISS nor classifier flagged SQL injection. "
            f"FAISS matches={len(faiss_result.matches)}, label={clf_result.label}"
        )

    def test_command_injection_flagged(self):
        """os.system with user input should be VULNERABLE."""
        code = """def ping(host):
    import os
    os.system("ping -c 1 " + host)"""
        result = check_vulnerability_patterns(code, "ping", "net.py")
        assert result.label == "VULNERABLE"

    def test_md5_password_flagged(self):
        """
        MD5 for passwords is a known vulnerability.
        Week 3 FAISS scanner always catches it. Full Kaggle model will too.
        """
        from tools.vulnerability_scanner import scan_for_vulnerabilities
        code = """def hash_pw(pw):
    import hashlib
    return hashlib.md5(pw.encode()).hexdigest()"""
        faiss_result = scan_for_vulnerabilities(code, "hash_pw", "auth.py")
        faiss_flagged = bool(faiss_result.matches)
        clf_result = check_vulnerability_patterns(code, "hash_pw", "auth.py")
        assert faiss_flagged or clf_result.label == "VULNERABLE"

    def test_pickle_loads_flagged(self):
        """pickle.loads on untrusted data should be VULNERABLE."""
        code = """def load_session(data):
    import pickle
    return pickle.loads(data)"""
        result = check_vulnerability_patterns(code, "load_session", "session.py")
        assert result.label == "VULNERABLE"

    def test_method_is_string(self):
        """Method field should identify how the prediction was made."""
        code = "def add(a, b): return a + b"
        result = check_vulnerability_patterns(code, "add", "math.py")
        assert result.method in (
            "codebert_classifier",
            "faiss_fallback",
            "error",
        )


# ─── Fallback Behavior ────────────────────────────────────────────────────────

class TestFallbackBehavior:
    def test_fallback_still_returns_prediction(self):
        """Even without trained model, should return valid prediction."""
        code = """def get_user(username):
    cursor.execute("SELECT * FROM users WHERE name = '" + username + "'")"""
        result = check_vulnerability_patterns(code, "get_user", "db.py")
        assert result.label in ("VULNERABLE", "SAFE", "UNKNOWN")

    def test_no_exception_on_any_code(self):
        """Should never raise an exception, even on weird input."""
        weird_inputs = [
            "",
            "not valid python",
            "def f(): " + "x" * 5000,   # very long
            "λ = lambda: None",          # unicode
        ]
        for code in weird_inputs:
            try:
                result = check_vulnerability_patterns(code, "test", "test.py")
                assert isinstance(result, ClassifierPrediction)
            except Exception as e:
                pytest.fail(f"Raised exception on input '{code[:30]}': {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
