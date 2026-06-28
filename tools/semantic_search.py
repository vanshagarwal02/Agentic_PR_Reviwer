"""
semantic_search.py — Tool 3: Find similar functions in the codebase

WHY THIS EXISTS:
  Call graphs tell you about DIRECT relationships (A calls B).
  But sometimes bugs come from INDIRECT similarity — two functions that do
  similar things but in different parts of the codebase. If one has a bug,
  the other probably does too.

  Example: You fix an injection vulnerability in `validate_user_input()`.
  Semantic search finds `sanitize_form_data()` and `check_query_params()`
  which are structurally similar — the agent should flag those too.

HOW IT WORKS (two modes):
  Mode 1 — Local (no API key needed):
    Uses TF-IDF (Term Frequency - Inverse Document Frequency) vectorization.
    Treats each function's source code as a "document".
    Measures similarity by angle between high-dimensional vectors.
    Fast, works offline, no cost. Less accurate than embeddings.

  Mode 2 — Gemini Embeddings (requires GEMINI_API_KEY):
    Sends function source to Gemini's text-embedding-004 model.
    Gets back a 768-dimensional vector that captures MEANING, not just keywords.
    "validate_token" and "authenticate_user" would be similar even if they
    share no words, because they do conceptually similar things.

  The tool auto-detects which mode to use based on whether the key is set.

WHAT SIMILARITY SCORE MEANS:
  1.0 = identical
  0.8+ = very similar (likely same pattern)
  0.5–0.8 = related (similar domain)
  <0.5 = probably unrelated

INPUT:
  query_function: ChangedFunction from parse_diff
  repo_path: str — path to the cloned repo

OUTPUT:
  SemanticSearchResult with ranked list of similar functions
"""

import os
import sys
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# ── Path fix ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# ──────────────────────────────────────────────────────────────────────────────

import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from config import config


# ─── Tree-sitter Setup ────────────────────────────────────────────────────────

def _get_python_parser() -> Parser:
    """Build a tree-sitter Parser for Python. Handles both 0.21.x and 0.22+ APIs."""
    try:
        lang = Language(tspython.language(), "python")  # 0.21.x
    except TypeError:
        lang = Language(tspython.language())            # 0.22+
    parser = Parser()
    parser.set_language(lang)
    return parser


_PYTHON_PARSER = _get_python_parser()


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class SimilarFunction:
    """One result from the similarity search."""
    function_name: str      # name of the similar function
    file_path: str          # where it lives
    start_line: int
    end_line: int
    similarity_score: float  # 0.0 – 1.0
    source_code: str        # full source (so agent can show it in review)
    match_reason: str       # human-readable explanation of why it matched


@dataclass
class SemanticSearchResult:
    """Result for one query function — ranked list of similar functions."""
    query_function_name: str
    query_file_path: str
    mode_used: str                  # "tfidf" or "gemini_embeddings"
    similar_functions: list[SimilarFunction] = field(default_factory=list)
    total_functions_searched: int = 0


# ─── Text Extraction ──────────────────────────────────────────────────────────

def _tokenize_code(source_code: str) -> list[str]:
    """
    Convert source code into tokens for TF-IDF.

    We do basic preprocessing:
    1. Split camelCase and snake_case identifiers into words
       ("validateToken" → ["validate", "token"])
    2. Remove Python keywords and syntax noise
    3. Lowercase everything

    WHY: "validate_token" and "validateToken" and "ValidateToken" should
    all be treated as the same concept. TF-IDF is case-sensitive by default
    so we normalize.
    """
    # Split on non-alphanumeric characters
    raw_tokens = re.split(r'[^a-zA-Z0-9]', source_code)

    # Split camelCase: "validateToken" → ["validate", "Token"]
    expanded = []
    for token in raw_tokens:
        # Insert split points before uppercase letters
        parts = re.sub(r'([A-Z])', r' \1', token).split()
        expanded.extend(parts)

    # Lowercase and filter short/noisy tokens
    PYTHON_KEYWORDS = {
        "def", "return", "if", "else", "elif", "for", "while", "try",
        "except", "finally", "with", "as", "import", "from", "class",
        "self", "None", "True", "False", "and", "or", "not", "in",
        "is", "pass", "break", "continue", "raise", "yield", "lambda",
        "global", "nonlocal", "del", "assert", "async", "await",
    }

    tokens = []
    for t in expanded:
        t_lower = t.lower()
        # Keep tokens that are at least 3 chars and not pure keywords
        if len(t_lower) >= 3 and t_lower not in PYTHON_KEYWORDS:
            tokens.append(t_lower)

    return tokens


# ─── TF-IDF Implementation ────────────────────────────────────────────────────
# We implement a minimal TF-IDF from scratch (no sklearn) so we have zero
# extra dependencies and you understand exactly what it does.

def _compute_tf(tokens: list[str]) -> dict[str, float]:
    """
    TF = Term Frequency, using SUBLINEAR scaling.

    Raw TF: tf(w) = count(w) / total_words
    Problem: a parameter name like `text_string` appears 5× in the function body
    (once in the def line, 3× in assignments, once in return). Raw TF gives it
    5× the weight of `strip` which appears once — even though both are equally
    important structurally. This buries shared structural tokens like `strip`,
    `lower`, `replace` under a pile of repeated variable names.

    Sublinear TF: tf(w) = (1 + log(count(w))) / total_sublinear_weight
    This compresses 5 occurrences → 2.61, vs 1 occurrence → 1.0.
    Much better for code where variable names repeat by necessity.
    """
    if not tokens:
        return {}
    counts = Counter(tokens)
    # Apply sublinear scaling: 1 + log(count)
    sublinear = {word: 1.0 + math.log(count) for word, count in counts.items()}
    total = sum(sublinear.values())
    return {word: val / total for word, val in sublinear.items()}


def _compute_idf(all_token_sets: list[list[str]]) -> dict[str, float]:
    """
    IDF = Inverse Document Frequency = how RARE a word is across all documents.
    Common words ("get", "set") get low IDF. Rare words ("authentication") get high IDF.

    idf(word) = log(total_docs / (1 + docs_containing_word))
    +1 in denominator = Laplace smoothing (avoids division by zero)
    """
    n_docs = len(all_token_sets)
    doc_freq: dict[str, int] = {}

    for token_set in all_token_sets:
        for word in set(token_set):  # set() so each doc counts once per word
            doc_freq[word] = doc_freq.get(word, 0) + 1

    idf = {}
    for word, freq in doc_freq.items():
        idf[word] = math.log(n_docs / (1 + freq))
    return idf


def _tfidf_vector(tf: dict[str, float], idf: dict[str, float]) -> dict[str, float]:
    """TF-IDF vector = element-wise multiplication of TF and IDF."""
    return {word: tf_val * idf.get(word, 0.0) for word, tf_val in tf.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """
    Cosine similarity between two sparse vectors (represented as dicts).

    cos(θ) = (A · B) / (|A| × |B|)

    Range: 0.0 (completely different) to 1.0 (identical direction).

    WHY COSINE AND NOT EUCLIDEAN DISTANCE:
      Euclidean distance would penalize long functions (more words = bigger vector).
      Cosine only cares about the DIRECTION (what words, not how many times),
      so a 10-line and 100-line function doing the same thing score high similarity.
    """
    # Dot product — only compute for words that appear in both
    dot_product = sum(
        vec_a[word] * vec_b.get(word, 0.0)
        for word in vec_a
    )

    # Magnitudes
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)


# ─── Function Scanning ────────────────────────────────────────────────────────

@dataclass
class _FunctionRecord:
    """Internal struct used during scanning. Not exposed to users."""
    name: str
    file_path: str
    start_line: int
    end_line: int
    source: str


def _scan_repo_for_functions(repo_path: str) -> list[_FunctionRecord]:
    """
    Walk the repo and collect (name, file, source) for every Python function.
    Same skip-list as call_graph.py — ignore venv, __pycache__, etc.
    """
    SKIP_DIRS = {
        "venv", ".venv", "env", "__pycache__", ".git",
        "node_modules", "dist", "build", ".tox", "eggs",
        ".eggs", "htmlcov", ".mypy_cache", ".pytest_cache",
    }

    records: list[_FunctionRecord] = []

    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, repo_path)

            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
                funcs = _extract_functions_from_file(source, rel_path)
                records.extend(funcs)
            except Exception:
                pass  # skip unparseable files silently

    return records


def _extract_functions_from_file(source_code: str, file_path: str) -> list[_FunctionRecord]:
    """Parse a Python file and return all functions with their source."""
    source_bytes = source_code.encode("utf-8")
    tree = _PYTHON_PARSER.parse(source_bytes)
    source_lines = source_code.splitlines()
    records = []
    _collect_fn_records(tree.root_node, source_bytes, source_lines, file_path, records)
    return records


def _collect_fn_records(node, source_bytes, source_lines, file_path, results):
    """Walk AST and collect FunctionRecord for each function definition."""
    if node.type in ("function_definition", "decorated_definition"):
        func_node = node
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type == "function_definition":
                    func_node = child
                    break

        # Get name
        name = None
        for child in func_node.children:
            if child.type == "identifier":
                name = source_bytes[child.start_byte:child.end_byte].decode("utf-8")
                break

        if name:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            func_source = "\n".join(source_lines[start_line - 1: end_line])
            results.append(_FunctionRecord(
                name=name,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                source=func_source,
            ))

    for child in node.children:
        _collect_fn_records(child, source_bytes, source_lines, file_path, results)


# ─── Gemini Embeddings (Optional) ─────────────────────────────────────────────

def _get_gemini_embedding(text: str) -> Optional[list[float]]:
    """
    Get a 768-dimensional embedding vector from Gemini's text-embedding-004 model.

    Returns None if:
    - GEMINI_API_KEY is not configured
    - API call fails (rate limit, network error, etc.)

    WHY text-embedding-004 OVER OLDER MODELS:
      It's specifically trained on code + natural language mixed content.
      It understands that "authenticate" and "verify_identity" are semantically close.
      Old models like text-embedding-002 treat them as completely different strings.
    """
    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "your_gemini_key_here":
        return None

    try:
        import google.genai as genai
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text,
            config={"task_type": "retrieval_document"},
        )
        return result.embeddings[0].values
    except ImportError:
        return None  # google-genai not installed yet
    except Exception:
        return None  # API error — fall back to TF-IDF silently


def _cosine_similarity_dense(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two dense (non-sparse) vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _match_reason(score: float, query_name: str, result_name: str) -> str:
    """Generate a human-readable reason for why this function matched."""
    if score >= 0.9:
        return "Near-identical structure and vocabulary"
    elif score >= 0.75:
        return "Very similar logic pattern — likely same kind of operation"
    elif score >= 0.6:
        return "Similar vocabulary and code structure"
    elif score >= 0.5:
        return "Related domain — overlapping concepts"
    else:
        return "Weak structural similarity"


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def semantic_search(
    query_source: str,
    query_name: str,
    query_file: str,
    repo_path: str,
    top_k: int = 5,
    min_similarity: float = 0.15,
) -> SemanticSearchResult:
    """
    Find the top-k most similar functions to the query function in the repo.

    Args:
        query_source: Full source code of the changed function (from ChangedFunction.full_source)
        query_name: Function name (for display + to exclude self from results)
        query_file: File path of the query function (to exclude exact self-match)
        repo_path: Path to the local repo clone
        top_k: How many results to return (default 5)
        min_similarity: Minimum score to include in results (0.0–1.0)

    Returns:
        SemanticSearchResult with ranked similar functions

    Example:
        from tools.semantic_search import semantic_search
        result = semantic_search(
            query_source=changed_fn.full_source,
            query_name=changed_fn.function_name,
            query_file=changed_fn.file_path,
            repo_path="/path/to/repo",
        )
        for fn in result.similar_functions:
            print(f"{fn.function_name} ({fn.file_path}): {fn.similarity_score:.2f}")
    """
    result = SemanticSearchResult(
        query_function_name=query_name,
        query_file_path=query_file,
        mode_used="tfidf",  # default, may upgrade to gemini below
    )

    # Step 1: Collect all functions in the repo
    all_functions = _scan_repo_for_functions(repo_path)
    result.total_functions_searched = len(all_functions)

    # Exclude the query function itself from results
    candidates = [
        fn for fn in all_functions
        if not (fn.name == query_name and fn.file_path == query_file)
    ]

    if not candidates:
        return result

    # Step 2: Try Gemini embeddings first (better quality)
    query_embedding = _get_gemini_embedding(query_source)

    if query_embedding is not None:
        result.mode_used = "gemini_embeddings"
        similarities = []

        for fn in candidates:
            fn_embedding = _get_gemini_embedding(fn.source)
            if fn_embedding is None:
                continue
            score = _cosine_similarity_dense(query_embedding, fn_embedding)
            if score >= min_similarity:
                similarities.append((fn, score))

    else:
        # Step 3: Fall back to TF-IDF
        result.mode_used = "tfidf"

        # Tokenize all functions
        query_tokens = _tokenize_code(query_source)
        all_token_sets = [_tokenize_code(fn.source) for fn in candidates]

        # Build IDF from all documents (including the query)
        all_docs = [query_tokens] + all_token_sets
        idf = _compute_idf(all_docs)

        # Build query vector
        query_tf = _compute_tf(query_tokens)
        query_vec = _tfidf_vector(query_tf, idf)

        similarities = []
        for fn, tokens in zip(candidates, all_token_sets):
            fn_tf = _compute_tf(tokens)
            fn_vec = _tfidf_vector(fn_tf, idf)
            score = _cosine_similarity(query_vec, fn_vec)
            if score >= min_similarity:
                similarities.append((fn, score))

    # Step 4: Deduplicate by (name, file_path) — keep highest score for each
    # This avoids showing the same function twice (decorated + bare definition)
    seen: dict[tuple[str, str], float] = {}
    deduped: list[tuple[_FunctionRecord, float]] = []
    for fn, score in similarities:
        key = (fn.name, fn.file_path)
        if key not in seen or score > seen[key]:
            seen[key] = score
            deduped.append((fn, score))

    deduped.sort(key=lambda x: x[1], reverse=True)
    top_results = deduped[:top_k]

    # Step 5: Build output objects
    for fn, score in top_results:
        result.similar_functions.append(SimilarFunction(
            function_name=fn.name,
            file_path=fn.file_path,
            start_line=fn.start_line,
            end_line=fn.end_line,
            similarity_score=round(score, 4),
            source_code=fn.source,
            match_reason=_match_reason(score, query_name, fn.name),
        ))

    return result


# ─── Pretty Print ─────────────────────────────────────────────────────────────

def print_semantic_search_result(result: SemanticSearchResult) -> None:
    """Human-readable output for development/debugging."""
    print(f"\n{'='*60}")
    print(f"Semantic Search — query: {result.query_function_name}")
    print(f"Mode: {result.mode_used} | Searched: {result.total_functions_searched} functions")
    print(f"Results: {len(result.similar_functions)} matches above threshold")

    for i, fn in enumerate(result.similar_functions, 1):
        bar = "█" * int(fn.similarity_score * 20)
        print(f"\n  {i}. {fn.function_name}  [{fn.file_path}:{fn.start_line}]")
        print(f"     Score: {fn.similarity_score:.3f}  {bar}")
        print(f"     Reason: {fn.match_reason}")
        # Show first 2 lines of source as a preview
        preview = "\n       ".join(fn.source_code.splitlines()[:2])
        print(f"     Preview: {preview[:120]}")

    print(f"\n{'='*60}")


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 tools/semantic_search.py <repo_path> <function_name>")
        print("Example: python3 tools/semantic_search.py . parse_diff")
        sys.exit(1)

    repo_path = sys.argv[1]
    func_name = sys.argv[2]

    # Find the function in the repo to use as query
    print(f"Scanning {repo_path} for '{func_name}'...")
    all_fns = _scan_repo_for_functions(repo_path)
    query_fn = next((f for f in all_fns if f.name == func_name), None)

    if not query_fn:
        print(f"Function '{func_name}' not found in {repo_path}")
        print(f"Available: {[f.name for f in all_fns[:20]]}")
        sys.exit(1)

    result = semantic_search(
        query_source=query_fn.source,
        query_name=query_fn.name,
        query_file=query_fn.file_path,
        repo_path=repo_path,
    )
    print_semantic_search_result(result)
