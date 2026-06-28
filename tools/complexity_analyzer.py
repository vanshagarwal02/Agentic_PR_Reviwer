"""
complexity_analyzer.py — Tool 2: Measure code complexity using tree-sitter AST

WHY COMPLEXITY MATTERS FOR CODE REVIEW:
  - High cyclomatic complexity (>10) = hard to test, likely has hidden bugs
  - Deep nesting (>4 levels) = hard to read, often signals a design issue
  - Very long functions (>50 lines) = violates single-responsibility principle
  The agent uses these metrics to JUSTIFY a "needs refactoring" comment
  with specific numbers, not just opinions.

WHAT IS CYCLOMATIC COMPLEXITY?
  Invented by Thomas McCabe in 1976. It counts the number of independent
  paths through a function. Every branch (if, elif, for, while, and, or,
  try, except) adds 1 to the complexity. Base complexity = 1.

  Example:
      def foo(x, y):        # complexity = 1 (base)
          if x > 0:         # +1 → 2
              if y > 0:     # +1 → 3
                  return x
          for i in range(y): # +1 → 4
              pass
          return 0
  Final cyclomatic complexity = 4.

  Interpretation:
    1-5   → Simple, easy to test
    6-10  → Moderate, still manageable
    11-15 → Complex — consider refactoring
    >15   → Very complex — high bug risk, hard to test
"""

from dataclasses import dataclass
from typing import Optional
import tree_sitter_python as tspython
import tree_sitter_c as tsc
from tree_sitter import Language, Parser

# Reuse the language parsers from parse_diff (avoid rebuilding grammars)
# We rebuild here to keep this module self-contained (simpler imports)


# ─── Data Class ───────────────────────────────────────────────────────────────

@dataclass
class ComplexityResult:
    """Complexity metrics for a single function."""
    function_name: str
    language: str
    line_count: int                 # total lines in function body
    cyclomatic_complexity: int      # number of decision paths (see above)
    max_nesting_depth: int          # deepest indentation level
    parameter_count: int            # number of arguments
    has_recursion: bool             # does the function call itself?

    # ── Derived assessments ───────────────────────────────────────────────────
    @property
    def complexity_label(self) -> str:
        """Human-readable complexity assessment."""
        cc = self.cyclomatic_complexity
        if cc <= 5:
            return "LOW"
        elif cc <= 10:
            return "MODERATE"
        elif cc <= 15:
            return "HIGH"
        else:
            return "VERY_HIGH"

    @property
    def needs_refactoring(self) -> bool:
        """True if any metric crosses a warning threshold."""
        return (
            self.cyclomatic_complexity > 10
            or self.max_nesting_depth > 4
            or self.line_count > 50
            or self.parameter_count > 5
        )

    @property
    def review_comment(self) -> Optional[str]:
        """
        Generate a specific, actionable comment if the function is complex.
        Returns None if the function is simple (no comment needed).
        """
        issues = []
        if self.cyclomatic_complexity > 10:
            issues.append(
                f"cyclomatic complexity of {self.cyclomatic_complexity} "
                f"(threshold: 10) — consider splitting into smaller functions"
            )
        if self.max_nesting_depth > 4:
            issues.append(
                f"nesting depth of {self.max_nesting_depth} levels "
                f"(threshold: 4) — use early returns or extract helper functions"
            )
        if self.line_count > 50:
            issues.append(
                f"{self.line_count} lines (threshold: 50) — "
                f"violates single-responsibility principle"
            )
        if self.parameter_count > 5:
            issues.append(
                f"{self.parameter_count} parameters (threshold: 5) — "
                f"consider grouping related params into a dataclass/config object"
            )
        if self.has_recursion:
            issues.append(
                "recursive — ensure base case is reachable and "
                "consider adding a recursion depth guard"
            )

        if not issues:
            return None

        return (
            f"Function `{self.function_name}` has complexity concerns: "
            + "; ".join(issues) + "."
        )


# ─── Language Setup ───────────────────────────────────────────────────────────

def _get_parser(language: str) -> Optional[Parser]:
    """Build a tree-sitter Parser for the given language.

    NOTE on API version:
    tree-sitter 0.21.x requires Language(capsule, 'name').
    tree-sitter 0.22+ changed to Language(capsule) with no name.
    We detect which API is available at runtime so this works with both.
    """
    try:
        if language == "python":
            try:
                lang = Language(tspython.language(), "python")  # 0.21.x
            except TypeError:
                lang = Language(tspython.language())             # 0.22+
        elif language in ("c", "cpp"):
            try:
                lang = Language(tsc.language(), "c")             # 0.21.x
            except TypeError:
                lang = Language(tsc.language())                  # 0.22+
        else:
            return None
        parser = Parser()
        parser.set_language(lang)
        return parser
    except Exception as e:
        print(f"[WARNING] Could not build parser for {language}: {e}")
        return None


# ─── Cyclomatic Complexity ────────────────────────────────────────────────────

# These are the tree-sitter node types that represent branching / decision points.
# Each one adds 1 to cyclomatic complexity.
# WHY THESE NODES: They each represent a point where execution can take
# two different paths — exactly what cyclomatic complexity measures.

COMPLEXITY_NODES = {
    "python": {
        "if_statement",       # if / elif
        "elif_clause",        # elif specifically
        "for_statement",      # for loop
        "while_statement",    # while loop
        "except_clause",      # try/except — each except is a path
        "with_statement",     # context managers (minor branching)
        "boolean_operator",   # 'and' / 'or' in conditions add paths
        "conditional_expression",  # ternary: x if cond else y
        "comprehension",      # [x for x in ... if ...] if clause
    },
    "c": {
        "if_statement",
        "else_clause",
        "for_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "case_statement",
        "conditional_expression",  # ternary a ? b : c
        "binary_expression",       # && and || in C conditions
    },
    "cpp": {
        "if_statement",
        "else_clause",
        "for_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "case_statement",
        "conditional_expression",
        "binary_expression",
        "try_statement",
        "catch_clause",
    },
}


def _count_complexity(node, language: str) -> int:
    """
    Recursively count complexity-adding nodes in an AST subtree.
    Base complexity of 1 is added by the caller, not here.
    """
    target_types = COMPLEXITY_NODES.get(language, set())
    count = 0

    # DFS traversal — visit every node in the subtree
    def _visit(n):
        nonlocal count
        if n.type in target_types:
            count += 1
        for child in n.children:
            _visit(child)

    _visit(node)
    return count


# ─── Nesting Depth ────────────────────────────────────────────────────────────

# Nodes that introduce a new indentation/nesting level
NESTING_NODES = {
    "python": {
        "if_statement", "elif_clause", "else_clause",
        "for_statement", "while_statement",
        "try_statement", "except_clause", "finally_clause",
        "with_statement", "function_definition", "class_definition",
    },
    "c": {
        "if_statement", "else_clause",
        "for_statement", "while_statement", "do_statement",
        "switch_statement",
    },
    "cpp": {
        "if_statement", "else_clause",
        "for_statement", "while_statement", "do_statement",
        "switch_statement", "try_statement", "catch_clause",
    },
}


def _max_nesting_depth(node, language: str, current_depth: int = 0) -> int:
    """
    Walk the AST and return the maximum nesting depth reached.
    Depth increases by 1 every time we enter a block-level node.
    """
    target_types = NESTING_NODES.get(language, set())
    max_depth = current_depth

    for child in node.children:
        child_depth = current_depth + (1 if child.type in target_types else 0)
        max_depth = max(max_depth, _max_nesting_depth(child, language, child_depth))

    return max_depth


# ─── Parameter Count ──────────────────────────────────────────────────────────

def _count_parameters(func_node, language: str) -> int:
    """
    Count the number of parameters in a function definition.
    Handles Python (parameters node) and C/C++ (parameter_list node).
    """
    param_node_types = {
        "python": "parameters",
        "c": "parameter_list",
        "cpp": "parameter_list",
    }
    target = param_node_types.get(language, "parameters")

    for child in func_node.children:
        if child.type == target:
            # Count child nodes that are actual parameters (not commas/parens)
            PARAM_TYPES = {
                "python": {"identifier", "typed_parameter", "default_parameter",
                           "list_splat_pattern", "dictionary_splat_pattern"},
                "c": {"parameter_declaration"},
                "cpp": {"parameter_declaration"},
            }
            valid_types = PARAM_TYPES.get(language, {"identifier"})
            count = sum(1 for p in child.children if p.type in valid_types)
            # Python 'self' and 'cls' are not real parameters — subtract them
            if language == "python":
                for p in child.children:
                    if p.type == "identifier":
                        name_bytes = p.text  # tree-sitter stores text as bytes
                        if name_bytes in (b"self", b"cls"):
                            count -= 1
            return max(0, count)
    return 0


# ─── Recursion Detection ──────────────────────────────────────────────────────

def _detect_recursion(func_node, func_name: str, source_bytes: bytes) -> bool:
    """
    Check if a function calls itself (directly recursive).
    We look for any call_expression/call node where the function name
    matches the current function's name.

    WHY THIS MATTERS: Recursion without a guaranteed base case is a
    common source of stack overflow bugs. Worth flagging for review.
    """
    func_name_bytes = func_name.encode("utf-8")

    def _search(node) -> bool:
        # Python: call node with 'call' type
        # C/C++: 'call_expression'
        if node.type in ("call", "call_expression"):
            # The first child of a call is the function being called
            if node.children:
                callee = node.children[0]
                if source_bytes[callee.start_byte:callee.end_byte] == func_name_bytes:
                    return True
        return any(_search(child) for child in node.children)

    return _search(func_node)


# ─── Main Entry Point ──────────────────────────────────────────────────────────

def get_function_complexity(code_snippet: str, language: str = "python") -> ComplexityResult:
    """
    Analyse a single function's code and return complexity metrics.

    Args:
        code_snippet: The full source code of ONE function (just the function,
                      not the whole file). Include the def/function line.
        language: "python", "c", or "cpp"

    Returns:
        ComplexityResult with all metrics filled in.

    Usage:
        code = '''
        def process_auth(token, user_id, session, db, config, fallback):
            if token:
                if session.is_valid():
                    for perm in db.get_permissions(user_id):
                        if perm.active:
                            if config.strict_mode:
                                return True
            return False
        '''
        result = get_function_complexity(code, "python")
        print(result.cyclomatic_complexity)  # 5
        print(result.max_nesting_depth)      # 5
        print(result.needs_refactoring)      # True
    """
    parser = _get_parser(language)
    if parser is None:
        # Return a minimal result if we can't parse the language
        return ComplexityResult(
            function_name="<unknown>",
            language=language,
            line_count=len(code_snippet.splitlines()),
            cyclomatic_complexity=1,
            max_nesting_depth=0,
            parameter_count=0,
            has_recursion=False,
        )

    source_bytes = code_snippet.strip().encode("utf-8")
    tree = parser.parse(source_bytes)

    # Find the top-level function definition node
    func_node = _find_function_node(tree.root_node, language)
    if func_node is None:
        # Code snippet doesn't start with a function definition
        # (might be a class method passed without context — handle gracefully)
        return ComplexityResult(
            function_name="<anonymous>",
            language=language,
            line_count=len(code_snippet.splitlines()),
            cyclomatic_complexity=1,
            max_nesting_depth=0,
            parameter_count=0,
            has_recursion=False,
        )

    # Extract function name
    func_name = _extract_name(func_node, source_bytes, language)

    # Count lines (exclude blank lines and comment-only lines for cleaner metric)
    lines = [l for l in code_snippet.strip().splitlines() if l.strip()]
    line_count = len(lines)

    # Cyclomatic complexity: base 1 + count of branching nodes
    cc = 1 + _count_complexity(func_node, language)

    # Nesting depth: how many levels deep does control flow go?
    nesting = _max_nesting_depth(func_node, language)

    # Parameter count
    param_count = _count_parameters(func_node, language)

    # Recursion check
    has_recursion = _detect_recursion(func_node, func_name, source_bytes)

    return ComplexityResult(
        function_name=func_name,
        language=language,
        line_count=line_count,
        cyclomatic_complexity=cc,
        max_nesting_depth=nesting,
        parameter_count=param_count,
        has_recursion=has_recursion,
    )


def _find_function_node(root_node, language: str):
    """Walk the root to find the first function definition node."""
    FUNC_TYPES = {
        "python": {"function_definition", "decorated_definition"},
        "c": {"function_definition"},
        "cpp": {"function_definition"},
    }
    targets = FUNC_TYPES.get(language, {"function_definition"})

    def _search(node):
        if node.type in targets:
            return node
        for child in node.children:
            result = _search(child)
            if result:
                return result
        return None

    return _search(root_node)


def _extract_name(func_node, source_bytes: bytes, language: str) -> str:
    """Get the function name from a function definition node."""
    # For decorated Python functions, drill into the inner function_definition
    actual = func_node
    if func_node.type == "decorated_definition":
        for child in func_node.children:
            if child.type == "function_definition":
                actual = child
                break

    for child in actual.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
    return "<unknown>"


# ─── Batch Analysis ───────────────────────────────────────────────────────────

def analyze_functions_complexity(
    functions: list[tuple[str, str]]  # list of (code_snippet, language)
) -> list[ComplexityResult]:
    """
    Analyse multiple functions at once.
    Used by the agent to process all changed functions from parse_diff() in one shot.

    Args:
        functions: list of (code_snippet, language) tuples

    Returns:
        list of ComplexityResult, one per function
    """
    return [get_function_complexity(code, lang) for code, lang in functions]


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with a deliberately complex function to verify the metrics
    test_code = '''
def process_payment(user_id, amount, currency, card_token, db, config, logger):
    """Deliberately complex function for testing."""
    if amount <= 0:
        raise ValueError("Amount must be positive")

    if currency not in config.supported_currencies:
        if config.allow_conversion:
            amount = convert_currency(amount, currency, "USD")
            currency = "USD"
        else:
            return {"status": "error", "reason": "unsupported_currency"}

    try:
        user = db.get_user(user_id)
        if user is None:
            logger.warning(f"User {user_id} not found")
            return {"status": "error", "reason": "user_not_found"}

        for limit in user.spending_limits:
            if limit.currency == currency and amount > limit.value:
                if not user.has_override:
                    return {"status": "error", "reason": "limit_exceeded"}

        result = payment_gateway.charge(card_token, amount, currency)
        if result.success:
            db.record_transaction(user_id, amount, currency, result.transaction_id)
            logger.info(f"Payment success: {result.transaction_id}")
            return {"status": "success", "transaction_id": result.transaction_id}
        else:
            logger.error(f"Payment failed: {result.error}")
            return {"status": "error", "reason": result.error}

    except DatabaseError as e:
        logger.error(f"DB error during payment: {e}")
        return {"status": "error", "reason": "database_error"}
    except GatewayError as e:
        logger.error(f"Gateway error: {e}")
        return {"status": "error", "reason": "gateway_error"}
'''

    result = get_function_complexity(test_code, "python")
    print(f"\nFunction: {result.function_name}")
    print(f"Lines: {result.line_count}")
    print(f"Cyclomatic Complexity: {result.cyclomatic_complexity} ({result.complexity_label})")
    print(f"Max Nesting Depth: {result.max_nesting_depth}")
    print(f"Parameters: {result.parameter_count}")
    print(f"Has Recursion: {result.has_recursion}")
    print(f"Needs Refactoring: {result.needs_refactoring}")
    print(f"\nReview Comment:\n{result.review_comment}")
