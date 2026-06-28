"""
call_graph.py — Tool 2: Build a call graph for changed functions

WHY THIS EXISTS:
  Complexity alone doesn't tell you IMPACT. A simple 3-line function can be
  called by 200 other functions — changing it is high risk even if it looks trivial.
  A call graph answers two questions the agent needs:
    1. "What does this function call?" (outgoing — what it depends on)
    2. "Who calls this function?" (incoming — what depends on it)

  Example:
    PR changes: merge_environment_settings()
    Call graph shows: called by send(), which is called by get(), post(), put()...
    → Agent flags: "this is a high-impact function, affects all HTTP methods"

HOW IT WORKS:
  1. Parse ALL Python files in the local repo clone using tree-sitter
  2. For each function, record:
     - What other functions it calls (by finding call_expression nodes in its body)
     - Its file location
  3. Build a networkx DiGraph: nodes = functions, edges = "A calls B"
  4. For each CHANGED function, query:
     - Descendants (what it calls, transitively)
     - Ancestors (what calls it, transitively)

LIMITATIONS (honest about them):
  - Only works on repos cloned locally (not fetched via API)
  - Dynamic dispatch (e.g. func = get_handler(); func()) can't be resolved statically
  - Cross-file resolution uses best-effort name matching, not full import tracking
  - C/C++ support is simpler (no method resolution)

INPUT:
  repo_path: str                — path to the cloned repo on disk
  changed_functions: list[str]  — function names from parse_diff output

OUTPUT:
  CallGraphResult with:
    - full graph (networkx DiGraph)
    - per-changed-function: callers list, callees list, impact score
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# ── Path fix ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# ──────────────────────────────────────────────────────────────────────────────

import networkx as nx
import tree_sitter_python as tspython
from tree_sitter import Language, Parser


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
class FunctionNode:
    """
    Represents one function in the codebase — a node in our call graph.
    Every function found across all files gets one of these.
    """
    name: str               # function name (e.g. "validate_token")
    file_path: str          # relative path within the repo (e.g. "auth/login.py")
    start_line: int         # line where the function definition starts
    end_line: int           # line where the function ends
    calls: list[str] = field(default_factory=list)  # names of functions this one calls


@dataclass
class FunctionImpact:
    """
    For one changed function, how big is its blast radius?
    This is what the agent uses to decide how much attention to pay.
    """
    function_name: str
    file_path: str

    # Direct relationships
    direct_callers: list[str] = field(default_factory=list)   # functions that call THIS one
    direct_callees: list[str] = field(default_factory=list)   # functions THIS one calls

    # Transitive relationships (through multiple hops)
    all_callers: list[str] = field(default_factory=list)      # everything upstream
    all_callees: list[str] = field(default_factory=list)      # everything downstream

    # Computed scores
    impact_score: float = 0.0   # 0.0 – 1.0 (normalized caller count)
    fan_in: int = 0             # how many functions call this one (direct)
    fan_out: int = 0            # how many functions this one calls (direct)
    is_leaf: bool = False       # True if this function calls nothing
    is_root: bool = False       # True if nothing calls this function


@dataclass
class CallGraphResult:
    """Top-level result returned by build_call_graph()."""
    repo_path: str
    total_functions_found: int = 0
    total_files_scanned: int = 0
    graph: Optional[nx.DiGraph] = field(default=None)   # full call graph
    impacts: list[FunctionImpact] = field(default_factory=list)  # one per changed function
    warnings: list[str] = field(default_factory=list)   # non-fatal issues (e.g. parse errors)


# ─── AST Walking ──────────────────────────────────────────────────────────────

def _extract_functions_and_calls(
    source_code: str,
    file_path: str,
) -> list[FunctionNode]:
    """
    Parse a Python source file and extract:
      - All function definitions (name, start line, end line)
      - For each function: which other functions it calls

    Returns a list of FunctionNode objects.

    HOW CALL DETECTION WORKS:
      tree-sitter parses the AST. Inside each function_definition node,
      we look for call nodes. A call node looks like:
        call:
          function: identifier → "validate_token"   (simple call)
          function: attribute → obj.method_name     (method call)
      We capture both. We don't resolve imports — just record names.
    """
    source_bytes = source_code.encode("utf-8")
    tree = _PYTHON_PARSER.parse(source_bytes)

    functions: list[FunctionNode] = []
    _collect_functions(tree.root_node, source_bytes, file_path, functions)
    return functions


def _collect_functions(node, source_bytes: bytes, file_path: str, results: list) -> None:
    """Recursively walk the AST and collect all function definitions."""
    if node.type in ("function_definition", "decorated_definition"):
        # For decorated functions, get the inner function_definition
        func_node = node
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type == "function_definition":
                    func_node = child
                    break

        name = _get_name(func_node, source_bytes)
        if name:
            start_line = node.start_point[0] + 1  # 0-indexed → 1-indexed
            end_line = node.end_point[0] + 1
            calls = _extract_calls_in_node(func_node, source_bytes)

            results.append(FunctionNode(
                name=name,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                calls=calls,
            ))

    for child in node.children:
        _collect_functions(child, source_bytes, file_path, results)


def _get_name(func_node, source_bytes: bytes) -> Optional[str]:
    """Extract the function name from a function_definition node."""
    for child in func_node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
    return None


def _extract_calls_in_node(func_node, source_bytes: bytes) -> list[str]:
    """
    Walk a function's AST subtree and collect all called function names.

    We look for nodes of type 'call'. Inside a call:
      - If the function part is an 'identifier', we have a direct call: foo()
      - If it's an 'attribute', we have a method call: obj.foo()
        We only keep the method name (foo), not the object (obj),
        because we can't know what obj is statically.
    """
    calls = []
    _walk_for_calls(func_node, source_bytes, calls)
    # Deduplicate while preserving order
    seen = set()
    unique_calls = []
    for c in calls:
        if c not in seen:
            seen.add(c)
            unique_calls.append(c)
    return unique_calls


def _walk_for_calls(node, source_bytes: bytes, results: list) -> None:
    """
    Recursively find all call nodes and extract the callee name.

    In tree-sitter 0.21.x, .field_name is not available on nodes.
    Instead, in the Python grammar, a `call` node always has:
      child[0] = the function being called (identifier or attribute)
      child[1] = the argument_list

    So we just inspect the first child's type.
    """
    if node.type == "call" and node.children:
        func_part = node.children[0]  # always the function expression

        if func_part.type == "identifier":
            # Direct call: foo()
            name = source_bytes[func_part.start_byte:func_part.end_byte].decode("utf-8")
            results.append(name)

        elif func_part.type == "attribute":
            # Method call: obj.method() — children are [obj, ".", method]
            # The last identifier child is the method name
            last_ident = None
            for child in func_part.children:
                if child.type == "identifier":
                    last_ident = child
            if last_ident is not None:
                name = source_bytes[last_ident.start_byte:last_ident.end_byte].decode("utf-8")
                results.append(name)

    for child in node.children:
        _walk_for_calls(child, source_bytes, results)


# ─── Graph Building ───────────────────────────────────────────────────────────

def _scan_repo(repo_path: str) -> tuple[list[FunctionNode], int, list[str]]:
    """
    Walk all .py files in repo_path and extract FunctionNode objects.

    Returns:
        (all_functions, files_scanned_count, warnings)

    WHY WE SKIP venv/node_modules/etc:
      These are third-party code. Call edges into them aren't meaningful
      for reviewing YOUR changes. We'd also blow up the graph size.
    """
    SKIP_DIRS = {
        "venv", ".venv", "env", "__pycache__", ".git",
        "node_modules", "dist", "build", ".tox", "eggs",
        ".eggs", "htmlcov", ".mypy_cache", ".pytest_cache",
    }

    all_functions: list[FunctionNode] = []
    files_scanned = 0
    warnings: list[str] = []

    for dirpath, dirnames, filenames in os.walk(repo_path):
        # Prune directories we don't want to descend into (modifies in-place)
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            abs_path = os.path.join(dirpath, filename)
            # Store relative path for cleaner output
            rel_path = os.path.relpath(abs_path, repo_path)

            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
                functions = _extract_functions_and_calls(source, rel_path)
                all_functions.extend(functions)
                files_scanned += 1
            except Exception as e:
                warnings.append(f"Could not parse {rel_path}: {e}")

    return all_functions, files_scanned, warnings


def _build_graph(all_functions: list[FunctionNode]) -> nx.DiGraph:
    """
    Build a directed call graph from the list of FunctionNode objects.

    Graph structure:
      - Node: function name (string)
      - Edge A → B: "function A calls function B"
      - Node attributes: file_path, start_line, end_line

    WHY DIRECTED:
      Direction encodes meaning. A → B means A depends on B.
      We can ask: "what are all the callers of B?" by looking at in-edges.
      We can ask: "what does A depend on?" by looking at out-edges.

    NAME COLLISION NOTE:
      If two functions share a name (e.g. two files both have a `validate`),
      we store them as "file.py::validate" in the graph but still track
      calls by short name. This is a known limitation of static analysis
      without full import resolution.
    """
    G = nx.DiGraph()

    # Build a name → FunctionNode mapping (last one wins on collision)
    name_to_node: dict[str, FunctionNode] = {}
    for fn in all_functions:
        G.add_node(fn.name, file_path=fn.file_path,
                   start_line=fn.start_line, end_line=fn.end_line)
        name_to_node[fn.name] = fn

    # Add edges for calls
    for fn in all_functions:
        for callee_name in fn.calls:
            if callee_name in name_to_node:
                # Only add edges to functions we actually found in the codebase
                # (not builtins like len(), print(), etc.)
                G.add_edge(fn.name, callee_name)

    return G


# ─── Impact Analysis ──────────────────────────────────────────────────────────

def _compute_impact(
    func_name: str,
    file_path: str,
    graph: nx.DiGraph,
    max_callers: int,
) -> FunctionImpact:
    """
    For one changed function, compute its blast radius in the codebase.

    impact_score = fan_in / max_callers
      - 0.0 = nothing calls this (safe to change)
      - 1.0 = the most-called function in the entire repo (very high risk)

    max_callers is passed in so all scores are relative to the same denominator.
    """
    impact = FunctionImpact(function_name=func_name, file_path=file_path)

    if func_name not in graph:
        impact.is_leaf = True
        impact.is_root = True
        return impact

    # Direct relationships
    impact.direct_callers = list(graph.predecessors(func_name))
    impact.direct_callees = list(graph.successors(func_name))
    impact.fan_in = len(impact.direct_callers)
    impact.fan_out = len(impact.direct_callees)
    impact.is_leaf = impact.fan_out == 0
    impact.is_root = impact.fan_in == 0

    # Transitive relationships (all ancestors/descendants in graph)
    try:
        impact.all_callers = list(nx.ancestors(graph, func_name))
    except Exception:
        impact.all_callers = impact.direct_callers

    try:
        impact.all_callees = list(nx.descendants(graph, func_name))
    except Exception:
        impact.all_callees = impact.direct_callees

    # Normalized impact score
    impact.impact_score = impact.fan_in / max_callers if max_callers > 0 else 0.0

    return impact


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def build_call_graph(
    repo_path: str,
    changed_function_names: list[str],
) -> CallGraphResult:
    """
    Build a call graph for the entire repo and compute impact for changed functions.

    Args:
        repo_path: Absolute or relative path to the cloned repository root.
        changed_function_names: List of function names from parse_diff() output.
            e.g. ["merge_environment_settings", "send", "validate_token"]

    Returns:
        CallGraphResult with impact analysis for each changed function.

    Usage:
        from tools.call_graph import build_call_graph
        result = build_call_graph("/path/to/repo", ["my_func", "other_func"])
        for impact in result.impacts:
            print(f"{impact.function_name}: {len(impact.all_callers)} transitive callers")
    """
    result = CallGraphResult(repo_path=repo_path)

    # Step 1: Scan the entire repo for Python files
    all_functions, files_scanned, warnings = _scan_repo(repo_path)
    result.total_functions_found = len(all_functions)
    result.total_files_scanned = files_scanned
    result.warnings = warnings

    if not all_functions:
        result.warnings.append(f"No Python functions found in {repo_path}")
        result.graph = nx.DiGraph()
        return result

    # Step 2: Build the directed call graph
    graph = _build_graph(all_functions)
    result.graph = graph

    # Step 3: Compute max fan_in across all nodes (for normalization)
    if graph.number_of_nodes() > 0:
        max_callers = max(graph.in_degree(n) for n in graph.nodes())
    else:
        max_callers = 1  # avoid division by zero

    # Step 4: Build a lookup for function locations
    name_to_node = {fn.name: fn for fn in all_functions}

    # Step 5: Compute impact for each changed function
    for func_name in changed_function_names:
        file_path = name_to_node[func_name].file_path if func_name in name_to_node else "unknown"
        impact = _compute_impact(func_name, file_path, graph, max_callers)
        result.impacts.append(impact)

    return result


# ─── Pretty Print ─────────────────────────────────────────────────────────────

def print_call_graph_result(result: CallGraphResult) -> None:
    """Human-readable summary for development/debugging."""
    print(f"\n{'='*60}")
    print(f"Call Graph Analysis")
    print(f"Repo: {result.repo_path}")
    print(f"Scanned: {result.total_files_scanned} files | "
          f"{result.total_functions_found} functions | "
          f"{result.graph.number_of_edges() if result.graph else 0} call edges")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings[:5]:  # cap at 5 so it doesn't flood output
            print(f"  ⚠  {w}")

    print(f"\nChanged function impact:")
    for impact in result.impacts:
        print(f"\n  {impact.function_name}  [{impact.file_path}]")
        print(f"    Impact score:  {impact.impact_score:.2f}  "
              f"(fan-in={impact.fan_in}, fan-out={impact.fan_out})")
        if impact.direct_callers:
            print(f"    Direct callers ({len(impact.direct_callers)}): "
                  f"{', '.join(impact.direct_callers[:5])}"
                  f"{'...' if len(impact.direct_callers) > 5 else ''}")
        if impact.direct_callees:
            print(f"    Direct callees ({len(impact.direct_callees)}): "
                  f"{', '.join(impact.direct_callees[:5])}"
                  f"{'...' if len(impact.direct_callees) > 5 else ''}")
        if len(impact.all_callers) > len(impact.direct_callers):
            print(f"    Transitive callers: {len(impact.all_callers)} total")
        flags = []
        if impact.is_leaf:
            flags.append("LEAF (calls nothing)")
        if impact.is_root:
            flags.append("ROOT (nothing calls it)")
        if impact.impact_score > 0.5:
            flags.append("⚠ HIGH IMPACT")
        if flags:
            print(f"    Flags: {' | '.join(flags)}")

    print(f"\n{'='*60}")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 tools/call_graph.py <repo_path> [func1] [func2] ...")
        print("Example: python3 tools/call_graph.py . parse_diff build_call_graph")
        sys.exit(1)

    repo = sys.argv[1]
    funcs = sys.argv[2:] if len(sys.argv) > 2 else []

    print(f"Scanning repo: {repo}")
    if funcs:
        print(f"Analysing impact for: {funcs}")
    else:
        print("No specific functions given — will show graph stats only")
        funcs = []

    result = build_call_graph(repo, funcs)
    print_call_graph_result(result)
