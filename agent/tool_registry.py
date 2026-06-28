"""
tool_registry.py — Maps tool names to real Python functions.

The agent selects tools by name (strings). This module executes them,
handles failures gracefully, and formats results as human-readable
strings for the synthesis prompt.

All tools run LOCALLY — no API calls, no rate limits.
"""

import logging
import os
import sys
import traceback

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger("ai_reviewer.tool_registry")

# ─── Tool Executors ───────────────────────────────────────────────────────────
# Each function takes (function_source, function_name, file_path, repo_path)
# and returns a human-readable string summary for the synthesis prompt.

def _run_scan_for_vulnerabilities(source: str, name: str, file_path: str, repo_path: str = None) -> str:
    from tools.vulnerability_scanner import scan_for_vulnerabilities
    result = scan_for_vulnerabilities(source, name, file_path)
    if not result.scanned:
        return f"{name}: scan failed - {result.scan_error}"
    if not result.matches:
        return f"{name}: No vulnerability patterns matched (checked {result.functions_in_index} patterns)."
    lines = [f"{name}: {len(result.matches)} vulnerability pattern(s) matched:"]
    for m in result.matches:
        lines.append(f"  [{m.severity}] {m.cwe} — {m.name} (similarity={m.similarity_score:.3f})")
        lines.append(f"    Description: {m.description}")
        lines.append(f"    Fix: {m.fix}")
    return "\n".join(lines)


def _run_check_vulnerability_patterns(source: str, name: str, file_path: str, repo_path: str = None) -> str:
    from tools.vulnerability_classifier import check_vulnerability_patterns
    pred = check_vulnerability_patterns(source, name, file_path)
    method_note = "(fine-tuned BigVul classifier)" if not pred.fallback_used else "(FAISS fallback)"
    return (
        f"{name}: label={pred.label} confidence={pred.confidence:.3f} "
        f"P(vulnerable)={pred.vulnerable_probability:.3f} {method_note}"
    )


def _run_build_call_graph(source: str, name: str, file_path: str, repo_path: str = None) -> str:
    if not repo_path or not os.path.isdir(repo_path):
        return f"{name}: Call graph skipped — no repo available."
    from tools.call_graph import build_call_graph
    result = build_call_graph(repo_path, [name])
    if not result.impacts:
        return f"{name}: No call graph data (function not found in repo index)."
    imp = result.impacts[0]
    lines = [
        f"{name}: impact_score={imp.impact_score:.3f} fan_in={imp.fan_in} fan_out={imp.fan_out}",
        f"  Direct callers: {imp.direct_callers or 'none'}",
        f"  Direct callees: {imp.direct_callees or 'none'}",
        f"  Transitive callers: {len(imp.all_callers)}",
        f"  High impact: {imp.impact_score > 0.5}",
    ]
    return "\n".join(lines)


def _run_semantic_search(source: str, name: str, file_path: str, repo_path: str = None) -> str:
    if not repo_path or not os.path.isdir(repo_path):
        return f"{name}: Semantic search skipped — no repo available."
    from tools.semantic_search import semantic_search
    result = semantic_search(
        query_source=source,
        query_name=name,
        query_file=file_path,
        repo_path=repo_path,
        top_k=3,
    )
    if not result.similar_functions:
        return f"{name}: No similar functions found in codebase."
    lines = [f"{name}: {len(result.similar_functions)} similar function(s) found:"]
    for sf in result.similar_functions:
        lines.append(
            f"  {sf.function_name} in {sf.file_path} "
            f"(score={sf.similarity_score:.3f}, {sf.match_reason})"
        )
    return "\n".join(lines)


# ─── Registry ─────────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "scan_for_vulnerabilities":     _run_scan_for_vulnerabilities,
    "check_vulnerability_patterns": _run_check_vulnerability_patterns,
    "build_call_graph":             _run_build_call_graph,
    "semantic_search":              _run_semantic_search,
}

VALID_TOOLS = set(TOOL_REGISTRY.keys())


def execute_tool(
    tool_name: str,
    function_name: str,
    function_source: str,
    file_path: str,
    repo_path: str = None,
) -> str:
    """
    Execute one tool on one function. Returns a formatted result string.
    Never raises — always returns a string (error message on failure).
    """
    if tool_name not in TOOL_REGISTRY:
        return f"Unknown tool: {tool_name}"

    try:
        logger.info(f"Running tool={tool_name} fn={function_name}")
        result = TOOL_REGISTRY[tool_name](function_source, function_name, file_path, repo_path)
        logger.info(f"Tool={tool_name} fn={function_name} complete")
        return result
    except Exception as e:
        logger.warning(f"Tool={tool_name} fn={function_name} failed: {e}")
        return f"{function_name}: Tool {tool_name} failed — {str(e)}"


def execute_tools_for_plan(
    tools_plan: list[dict],  # [{tool, functions, reason}]
    function_sources: dict,  # {function_name: {source, file, start_line}}
    repo_path: str = None,
) -> dict:
    """
    Execute all tools selected by the agent's reasoning step.

    Args:
        tools_plan: list from Gemini's reasoning output
        function_sources: mapping of function name → source + metadata
        repo_path: path to cloned repo for call graph / semantic search

    Returns:
        dict mapping "tool_name:func_name" → result string
    """
    results = {}

    for tool_entry in tools_plan:
        tool_name = tool_entry.get("tool", "")
        functions  = tool_entry.get("functions", [])
        reason     = tool_entry.get("reason", "")

        if tool_name not in VALID_TOOLS:
            logger.warning(f"Agent requested unknown tool: {tool_name}")
            continue

        logger.info(f"Executing tool={tool_name} for functions={functions} reason='{reason}'")

        for fn_name in functions:
            fn_data = function_sources.get(fn_name)
            if not fn_data:
                # Try case-insensitive match
                fn_data = next(
                    (v for k, v in function_sources.items() if k.lower() == fn_name.lower()),
                    None,
                )
            if not fn_data or not fn_data.get("source"):
                results[f"{tool_name}:{fn_name}"] = (
                    f"{fn_name}: source not available for tool execution"
                )
                continue

            key = f"{tool_name}:{fn_name}"
            results[key] = execute_tool(
                tool_name=tool_name,
                function_name=fn_name,
                function_source=fn_data["source"],
                file_path=fn_data.get("file", "unknown"),
                repo_path=repo_path,
            )

    return results


def format_results_for_prompt(tool_results: dict) -> dict:
    """
    Group results by tool name for the synthesis prompt.
    Returns {tool_name: combined_result_string}
    """
    grouped = {}
    for key, result in tool_results.items():
        # key format: "tool_name:function_name"
        parts = key.split(":", 1)
        tool_name = parts[0] if len(parts) == 2 else "unknown"
        if tool_name not in grouped:
            grouped[tool_name] = []
        grouped[tool_name].append(result)

    return {tool: "\n".join(results) for tool, results in grouped.items()}
