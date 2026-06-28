# Week 2 — Diff Parsing, AST Analysis & Call Graphs

## What Week 2 Built

The code understanding layer. Before you can review code, you need to actually parse it — extract which functions changed, how complex they are, and what they call.

```
Raw GitHub diff (unified diff format)
        ↓
  parse_diff.py → ParsedPR + ChangedFunction objects
        ↓
  complexity_analyzer.py → cyclomatic complexity per function
        ↓
  call_graph.py → directed call graph (networkx DiGraph)
        ↓
  Structured data ready for agent (Week 5)
```

---

## Diff Parser (`tools/parse_diff.py`)

Unified diffs look like this:
```diff
@@ -10,6 +10,8 @@ def get_user(name):
-    query = "SELECT * FROM users WHERE name=" + name
+    query = "SELECT * FROM users WHERE name=?"
+    cursor.execute(query, (name,))
```

The parser extracts:
- Which files changed
- Which functions were modified (using tree-sitter AST to find function boundaries)
- The full source of each changed function
- Which specific lines changed

### Key dataclasses

```python
@dataclass
class ChangedFunction:
    file_path: str
    function_name: str
    language: str           # "python"
    start_line: int
    end_line: int
    changed_lines: list[int]
    full_source: str        # Complete function body
    context_before: str
    context_after: str
    is_new_file: bool

@dataclass
class ParsedPR:
    pr_number: int
    pr_title: str
    pr_description: str
    repo_full_name: str
    changed_functions: list[ChangedFunction]
    changed_files: list[str]
    total_additions: int
    total_deletions: int
```

---

## Complexity Analyzer (`tools/complexity_analyzer.py`)

Cyclomatic complexity = number of independent paths through a function. Computed via AST traversal — counts `if`, `for`, `while`, `except`, `and`, `or`, `with` nodes.

```
CC = 1 + (branches)
CC ≤ 5   → simple, easy to test
CC 6–10  → moderate
CC > 10  → complex, needs refactoring
```

```python
@dataclass
class ComplexityResult:
    function_name: str
    cyclomatic_complexity: int
    max_nesting_depth: int
    line_count: int
    needs_refactoring: bool   # CC > 10
```

Why this matters for security: complex functions (high CC) are statistically more likely to contain bugs and be harder to audit.

---

## Call Graph Builder (`tools/call_graph.py`)

Uses `networkx.DiGraph` to map which functions call which others.

```python
# Example output for a PR touching get_user():
{
  "get_user": ["execute_query", "log_access"],
  "execute_query": ["db_connect"],
}
```

Why this matters: a SQL injection in `execute_query` is critical if it's called by 10 public endpoints. If it's only called internally with sanitized inputs, it's lower risk. The agent uses this to understand blast radius.

---

## tree-sitter

All AST parsing uses `tree-sitter` — a fast, incremental parser that understands Python syntax properly (not regex hacks).

```python
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)
tree = parser.parse(source_code.encode())
```

---

## File Summary

```
tools/parse_diff.py          ← ParsedPR, ChangedFunction dataclasses + diff parser
tools/complexity_analyzer.py ← Cyclomatic complexity via AST
tools/call_graph.py          ← networkx DiGraph call graph builder
```

---

## Tests (`tests/test_week2.py`)

- Diff parser extracts correct function names from unified diffs
- Complexity calculator gives correct CC for known functions
- Call graph correctly identifies callee relationships
- Nesting depth calculated correctly
- `needs_refactoring` flag triggers at CC > 10
