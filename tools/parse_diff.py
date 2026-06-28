"""
parse_diff.py — Tool 1: Extract changed functions from a GitHub Pull Request

WHY THIS EXISTS:
  The agent needs to know WHAT changed, not just the raw diff text.
  A raw diff is a wall of +/- lines. This tool structures it into:
    - Which functions changed
    - In which files
    - At which line numbers
    - With surrounding context (so the LLM has enough to reason about it)

HOW IT WORKS (high level):
  1. Connect to GitHub API using your token
  2. Fetch the PR's list of changed files + the unified diff for each
  3. Parse the diff to find changed line ranges
  4. Use tree-sitter to parse the full file and find which function(s)
     contain those changed lines
  5. Return a clean list of ChangedFunction objects
"""

import re
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

# ── Path fix ──────────────────────────────────────────────────────────────────
# When this file is run directly (python3 tools/parse_diff.py), Python adds
# the tools/ directory to sys.path — NOT the project root. So 'import config'
# fails. We explicitly insert the project root (parent of this file's dir).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# ──────────────────────────────────────────────────────────────────────────────

import tree_sitter_python as tspython
import tree_sitter_c as tsc
from tree_sitter import Language, Parser
from github import Github, GithubException
from config import config


# ─── Data Classes ─────────────────────────────────────────────────────────────
# We use dataclasses (like structs) to give our data a clear shape.
# The agent will pass these objects between tools — having typed fields
# prevents subtle bugs ("wait, was it .line or .line_number?")

@dataclass
class ChangedHunk:
    """
    A 'hunk' is one contiguous block of changes in a diff.
    Example: lines 45-52 were modified = one hunk.
    A single function might have multiple hunks if you changed
    line 10 and line 80 separately.
    """
    old_start: int          # line number in the OLD file where hunk starts
    old_count: int          # how many lines this hunk spans in old file
    new_start: int          # line number in the NEW file where hunk starts
    new_count: int          # how many lines this hunk spans in new file
    added_lines: list[int] = field(default_factory=list)    # line numbers that were added
    removed_lines: list[int] = field(default_factory=list)  # line numbers that were removed
    diff_text: str = ""     # the raw +/- text for this hunk


@dataclass
class ChangedFunction:
    """
    One function (or method) that was touched by the PR.
    This is the primary unit the agent reasons about.
    """
    file_path: str          # e.g. "auth/login.py"
    function_name: str      # e.g. "validate_token"
    language: str           # "python", "c", "cpp", "unknown"
    start_line: int         # where the function definition starts in new file
    end_line: int           # where the function ends
    changed_lines: list[int] = field(default_factory=list)  # which lines inside it changed
    full_source: str = ""   # complete function body (for vulnerability classifier)
    context_before: str = "" # CONTEXT_LINES lines above function (for LLM)
    context_after: str = ""  # CONTEXT_LINES lines below function (for LLM)
    hunks: list[ChangedHunk] = field(default_factory=list)  # the actual diff hunks
    is_new_file: bool = False       # entire file is new (no old version)
    is_deleted_file: bool = False   # entire file was deleted


@dataclass
class ParsedPR:
    """Top-level result returned by parse_diff()"""
    pr_number: int
    pr_title: str
    pr_description: str
    repo_full_name: str     # e.g. "psf/requests"
    base_branch: str        # the branch being merged INTO (usually "main")
    head_branch: str        # the branch with the new changes
    changed_functions: list[ChangedFunction] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)   # all changed file paths
    total_additions: int = 0
    total_deletions: int = 0


# ─── Tree-sitter Setup ────────────────────────────────────────────────────────
# tree-sitter is a parsing library. Unlike Python's built-in `ast` module,
# it works for MULTIPLE languages and handles broken/partial code gracefully.
# We pre-build language objects once at module load (expensive operation).

def _build_languages() -> dict[str, tuple[Language, Parser]]:
    """
    Build tree-sitter Language + Parser pairs for each supported language.
    Returns a dict: {"python": (Language, Parser), "c": (Language, Parser)}

    NOTE: tree-sitter 0.21.x requires Language(capsule, 'name').
          tree-sitter 0.22+ uses Language(capsule) with no name.
          We try both to stay compatible.
    """
    langs = {}

    # Python
    try:
        try:
            py_lang = Language(tspython.language(), "python")   # 0.21.x
        except TypeError:
            py_lang = Language(tspython.language())              # 0.22+
        py_parser = Parser()
        py_parser.set_language(py_lang)
        langs["python"] = (py_lang, py_parser)
    except Exception as e:
        print(f"[WARNING] Could not load Python grammar: {e}")

    # C (we reuse for C++ since we're just doing basic function extraction)
    try:
        try:
            c_lang = Language(tsc.language(), "c")               # 0.21.x
        except TypeError:
            c_lang = Language(tsc.language())                    # 0.22+
        c_parser = Parser()
        c_parser.set_language(c_lang)
        langs["c"] = (c_lang, c_parser)
        langs["cpp"] = (c_lang, c_parser)   # C++ is close enough for extraction
    except Exception as e:
        print(f"[WARNING] Could not load C grammar: {e}")

    return langs


# Module-level constant — built once when the module is first imported
_LANGUAGE_PARSERS = _build_languages()


# ─── Language Detection ───────────────────────────────────────────────────────

def _detect_language(file_path: str) -> str:
    """Guess programming language from file extension."""
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    return {
        "py": "python",
        "c": "c",
        "cpp": "cpp", "cc": "cpp", "cxx": "cpp",
        "h": "c", "hpp": "cpp",
        "js": "javascript", "ts": "typescript",  # future support
    }.get(ext, "unknown")


# ─── Diff Parsing ─────────────────────────────────────────────────────────────

def _parse_unified_diff(patch_text: str) -> list[ChangedHunk]:
    """
    Parse a unified diff patch string into a list of ChangedHunk objects.

    A unified diff looks like:
        @@ -45,8 +45,9 @@          ← hunk header: old start/count, new start/count
         def validate_token(tok):  ← unchanged context line (space prefix)
        -    if tok is None:       ← removed line (- prefix)
        +    if tok is None or tok == "":  ← added line (+ prefix)

    The @@ line tells us WHERE in the file the change is.
    We parse each @@ block into a ChangedHunk.
    """
    if not patch_text:
        return []

    hunks: list[ChangedHunk] = []
    current_hunk: Optional[ChangedHunk] = None
    current_new_line = 0  # tracks current line number in the new file

    # Regex to match the hunk header line: @@ -old_start,old_count +new_start,new_count @@
    # The ,count part is optional (when count=1, GitHub omits it)
    hunk_header_re = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    for line in patch_text.splitlines():
        header_match = hunk_header_re.match(line)

        if header_match:
            # Save previous hunk
            if current_hunk:
                hunks.append(current_hunk)

            old_start = int(header_match.group(1))
            old_count = int(header_match.group(2) or 1)
            new_start = int(header_match.group(3))
            new_count = int(header_match.group(4) or 1)

            current_hunk = ChangedHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                diff_text=line + "\n",
            )
            current_new_line = new_start

        elif current_hunk is not None:
            current_hunk.diff_text += line + "\n"

            if line.startswith("+"):
                # Added line — record its line number in the new file
                current_hunk.added_lines.append(current_new_line)
                current_new_line += 1
            elif line.startswith("-"):
                # Removed line — line number in OLD file (useful for context)
                current_hunk.removed_lines.append(current_hunk.old_start +
                    len(current_hunk.removed_lines))
                # Don't increment current_new_line (removed lines don't exist in new file)
            else:
                # Context line (space prefix) — exists in both old and new
                current_new_line += 1

    if current_hunk:
        hunks.append(current_hunk)

    return hunks


# ─── Function Extraction via Tree-sitter ──────────────────────────────────────

def _extract_functions_from_source(
    source_code: str,
    language: str,
) -> list[tuple[str, int, int]]:
    """
    Use tree-sitter to parse source code and return all function definitions.
    Returns: list of (function_name, start_line, end_line)
    Lines are 1-indexed (matching what humans and GitHub use).

    WHY TREE-SITTER OVER PYTHON'S ast MODULE:
    - ast only works for Python; tree-sitter handles Python, C, C++, JS, etc.
    - tree-sitter handles syntax errors gracefully (partial parses still work)
    - tree-sitter is much faster for large files
    - We can use the SAME code path for all languages
    """
    if language not in _LANGUAGE_PARSERS:
        return []  # language not supported yet — silently skip

    _, parser = _LANGUAGE_PARSERS[language]

    # tree-sitter requires bytes, not str
    source_bytes = source_code.encode("utf-8")
    tree = parser.parse(source_bytes)

    functions = []
    _walk_for_functions(tree.root_node, source_bytes, language, functions)
    return functions


def _walk_for_functions(
    node,
    source_bytes: bytes,
    language: str,
    results: list,
) -> None:
    """
    Recursively walk the AST tree, collecting function/method definition nodes.

    WHY RECURSIVE WALK:
    Functions can be nested (closures, inner functions). We want ALL of them.
    tree-sitter nodes have a .children list — we walk depth-first.
    """
    # Node types that represent function definitions vary by language
    FUNCTION_NODE_TYPES = {
        "python": {"function_definition", "decorated_definition"},
        "c": {"function_definition"},
        "cpp": {"function_definition"},
    }

    target_types = FUNCTION_NODE_TYPES.get(language, set())

    if node.type in target_types:
        # For decorated functions in Python (e.g. @staticmethod def foo():)
        # the actual function_definition is a child of decorated_definition
        actual_func_node = node
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type == "function_definition":
                    actual_func_node = child
                    break

        name = _get_function_name(actual_func_node, source_bytes, language)
        if name:
            # tree-sitter uses 0-indexed rows, convert to 1-indexed lines
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            results.append((name, start_line, end_line))

    # Always recurse into children — we want nested functions too
    for child in node.children:
        _walk_for_functions(child, source_bytes, language, results)


def _get_function_name(func_node, source_bytes: bytes, language: str) -> Optional[str]:
    """Extract just the function name from a function definition AST node."""
    for child in func_node.children:
        if child.type == "identifier":
            # child.start_byte and end_byte give us the name's position in bytes
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
    return None


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def parse_diff(pr_url: str) -> ParsedPR:
    """
    Given a GitHub PR URL, return a ParsedPR with all changed functions.

    Args:
        pr_url: Full GitHub PR URL, e.g.
                "https://github.com/psf/requests/pull/6734"

    Returns:
        ParsedPR dataclass with .changed_functions list

    Raises:
        ValueError: if the URL doesn't look like a GitHub PR URL
        GithubException: if the token is invalid or PR doesn't exist
    """
    # ── Step 1: Parse the URL to get owner/repo/pr_number ─────────────────────
    # URL format: https://github.com/{owner}/{repo}/pull/{number}
    match = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url)
    if not match:
        raise ValueError(
            f"Invalid GitHub PR URL: {pr_url}\n"
            "Expected format: https://github.com/owner/repo/pull/123"
        )
    owner, repo_name, pr_number = match.group(1), match.group(2), int(match.group(3))

    # ── Step 2: Connect to GitHub API ─────────────────────────────────────────
    # PyGithub handles authentication, rate limiting headers, and pagination
    # automatically. Without a token, you get 60 requests/hour. With a token: 5000/hour.
    if not config.GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN not set in .env file")

    gh = Github(config.GITHUB_TOKEN)

    try:
        repo = gh.get_repo(f"{owner}/{repo_name}")
        pr = repo.get_pull(pr_number)
    except GithubException as e:
        raise GithubException(e.status, f"Could not fetch PR: {e.data}")

    # ── Step 3: Build the top-level ParsedPR object ────────────────────────────
    result = ParsedPR(
        pr_number=pr_number,
        pr_title=pr.title,
        # PR description can be None if the author left it blank
        pr_description=pr.body or "",
        repo_full_name=f"{owner}/{repo_name}",
        base_branch=pr.base.ref,
        head_branch=pr.head.ref,
        total_additions=pr.additions,
        total_deletions=pr.deletions,
    )

    # ── Step 4: Process each changed file ─────────────────────────────────────
    # pr.get_files() returns a paginated list of PullRequestFile objects.
    # Each file has: filename, patch (unified diff), status, additions, deletions
    for pr_file in pr.get_files():
        file_path = pr_file.filename
        result.changed_files.append(file_path)

        language = _detect_language(file_path)

        # Skip files we can't meaningfully parse (configs, docs, images, etc.)
        if language == "unknown":
            continue

        # ── Step 4a: Parse the diff to find changed line numbers ───────────────
        hunks = _parse_unified_diff(pr_file.patch or "")
        if not hunks:
            continue

        # Collect all line numbers that changed in the new file
        all_changed_lines: set[int] = set()
        for hunk in hunks:
            all_changed_lines.update(hunk.added_lines)

        # ── Step 4b: Fetch the full file content to extract function bodies ────
        # We need the full file to know where each function starts/ends.
        # pr.head.sha is the commit hash of the PR's latest commit.
        try:
            file_content_obj = repo.get_contents(file_path, ref=pr.head.sha)
            # GitHub API returns base64-encoded content — .decoded_content decodes it
            full_source = file_content_obj.decoded_content.decode("utf-8", errors="replace")
        except Exception:
            # File might be binary, too large, or deleted
            full_source = ""

        source_lines = full_source.splitlines()

        # ── Step 4c: Find which functions contain the changed lines ────────────
        all_functions = _extract_functions_from_source(full_source, language)

        for func_name, func_start, func_end in all_functions:
            # Check if any changed line falls within this function's range
            func_lines = set(range(func_start, func_end + 1))
            overlap = all_changed_lines & func_lines

            if not overlap:
                continue  # This function wasn't touched — skip it

            # Extract the function's source code (slice the lines array)
            # Lines are 1-indexed, Python lists are 0-indexed → subtract 1
            func_source_lines = source_lines[func_start - 1 : func_end]
            func_full_source = "\n".join(func_source_lines)

            # Extract context lines before and after the function
            # These help the LLM understand the surrounding code structure
            context_start = max(0, func_start - 1 - config.CONTEXT_LINES)
            context_end = min(len(source_lines), func_end + config.CONTEXT_LINES)

            context_before_lines = source_lines[context_start : func_start - 1]
            context_after_lines = source_lines[func_end : context_end]

            # Find which hunks overlap with this function
            relevant_hunks = [
                h for h in hunks
                if set(range(h.new_start, h.new_start + h.new_count)) & func_lines
            ]

            changed_func = ChangedFunction(
                file_path=file_path,
                function_name=func_name,
                language=language,
                start_line=func_start,
                end_line=func_end,
                changed_lines=sorted(overlap),
                full_source=func_full_source,
                context_before="\n".join(context_before_lines),
                context_after="\n".join(context_after_lines),
                hunks=relevant_hunks,
                is_new_file=(pr_file.status == "added"),
                is_deleted_file=(pr_file.status == "removed"),
            )
            result.changed_functions.append(changed_func)

        # ── Edge case: new file with no parseable functions (e.g. script) ──────
        # If the file is new but we found no functions, we still note the file
        # (the agent can handle this as a "module-level code" case in Week 5)

    gh.close()  # Release the connection pool
    return result


# ─── Pretty Print (for development/debugging) ─────────────────────────────────

def print_parsed_pr(parsed: ParsedPR) -> None:
    """Human-readable summary — use this during development to verify output."""
    print(f"\n{'='*60}")
    print(f"PR #{parsed.pr_number}: {parsed.pr_title}")
    print(f"Repo: {parsed.repo_full_name} | {parsed.base_branch} ← {parsed.head_branch}")
    print(f"Changes: +{parsed.total_additions} / -{parsed.total_deletions}")
    print(f"Changed files ({len(parsed.changed_files)}): {', '.join(parsed.changed_files[:5])}")
    print(f"Changed functions found: {len(parsed.changed_functions)}")
    print(f"{'='*60}")

    for cf in parsed.changed_functions:
        print(f"\n  [{cf.language.upper()}] {cf.file_path}::{cf.function_name}")
        print(f"    Lines {cf.start_line}–{cf.end_line} | Changed at: {cf.changed_lines}")
        if cf.is_new_file:
            print("    ⚡ NEW FILE")
        if cf.is_deleted_file:
            print("    🗑  DELETED FILE")
        print(f"    Hunks: {len(cf.hunks)}")


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python parse_diff.py <github-pr-url>")
        print("Example: python parse_diff.py https://github.com/psf/requests/pull/6734")
        sys.exit(1)

    pr_url = sys.argv[1]
    print(f"Fetching PR: {pr_url}")
    parsed = parse_diff(pr_url)
    print_parsed_pr(parsed)
