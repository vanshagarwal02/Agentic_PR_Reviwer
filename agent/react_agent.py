"""
react_agent.py — The ReAct agent core (Week 5).

Architecture: Two Gemini calls per PR, never more.
  1. REASON: Given PR context, select which tools to run.
  2. [Run tools locally — fast, free, no API calls]
  3. SYNTHESIZE: Given tool results, produce structured review.

Why only 2 calls?
  Free tier = 15 requests/minute, 1M tokens/day.
  Classic ReAct loops call the LLM after every single tool result.
  At 6 tools per PR that's 6+ calls just for tool selection.
  We batch it: ONE call selects ALL tools, ONE call synthesizes ALL results.
  This keeps us well within free tier even at 100 PRs/day.
"""

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import google.genai as genai
from google.genai import types as genai_types
from config import config
from agent.prompts import (
    SYSTEM_PROMPT,
    build_reasoning_prompt,
    build_synthesis_prompt,
)
from agent.tool_registry import (
    execute_tools_for_plan,
    format_results_for_prompt,
    VALID_TOOLS,
)

logger = logging.getLogger("ai_reviewer.agent")

# ─── Config ───────────────────────────────────────────────────────────────────

GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
RATE_LIMIT_SLEEP = 4   # seconds between Gemini calls (free tier safety)
MAX_RETRIES      = 3   # exponential backoff retries on 429
MAX_FUNCTIONS    = 10  # cap functions per PR to avoid token overload
SOURCE_PREVIEW_CHARS = 400  # chars of source shown to agent in reasoning step


# ─── Output Schema ────────────────────────────────────────────────────────────

@dataclass
class InlineComment:
    file: str
    line: int
    severity: str        # CRITICAL | HIGH | MEDIUM | LOW
    category: str        # SECURITY | CORRECTNESS | STYLE | COVERAGE
    comment: str
    suggestion: str = ""


@dataclass
class AgentReview:
    overall_verdict: str          # REQUEST_CHANGES | APPROVE | COMMENT
    risk_level: str               # LOW | MEDIUM | HIGH
    summary: str
    inline_comments: list[InlineComment] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    confidence: float = 0.0
    # Agent reasoning (logged, not returned to user)
    pr_type: str = ""
    reasoning: str = ""
    tool_plan: list[dict] = field(default_factory=list)
    raw_tool_results: dict = field(default_factory=dict)
    gemini_calls: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


# ─── Gemini Client ────────────────────────────────────────────────────────────

def _get_gemini_client():
    """Configure and return the Gemini client. Called lazily."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_key_here":
        raise ValueError(
            "GEMINI_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    return genai.Client(api_key=GEMINI_API_KEY)


def _call_gemini_with_retry(client, prompt: str, call_name: str) -> str:
    """
    Call Gemini with exponential backoff on rate limit errors.
    Returns the raw text response.
    """
    delay = RATE_LIMIT_SLEEP
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Gemini call: {call_name} (attempt {attempt})")
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            logger.info(f"Gemini call: {call_name} complete")
            return response.text
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                if attempt < MAX_RETRIES:
                    logger.warning(f"Rate limit hit on {call_name}, sleeping {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
                else:
                    raise RuntimeError(f"Gemini rate limit exceeded after {MAX_RETRIES} retries: {e}")
            else:
                raise


def _parse_json_response(text: str, call_name: str) -> dict:
    """
    Parse Gemini's JSON response robustly.
    Handles markdown code fences and trailing commas.
    """
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try removing trailing commas (common LLM mistake)
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {call_name} response: {e}\nRaw: {text[:500]}")
            raise ValueError(f"Could not parse {call_name} JSON response: {e}")


# ─── Context Builders ─────────────────────────────────────────────────────────

def _build_pr_context(parsed_pr, complexity_results: list) -> dict:
    """
    Build the PR context dict from a ParsedPR and complexity results.
    Caps at MAX_FUNCTIONS to avoid overloading the token window.
    """
    complexity_map = {r.function_name: r for r in (complexity_results or [])}

    changed_functions = []
    for cf in (parsed_pr.changed_functions or [])[:MAX_FUNCTIONS]:
        comp = complexity_map.get(cf.function_name)
        source = cf.full_source or ""
        changed_functions.append({
            "name":          cf.function_name,
            "file":          cf.file_path,
            "start_line":    cf.start_line,
            "end_line":      cf.end_line,
            "lines_changed": len(cf.changed_lines or []),
            "complexity":    comp.cyclomatic_complexity if comp else "?",
            "source_preview": source[:SOURCE_PREVIEW_CHARS] + ("..." if len(source) > SOURCE_PREVIEW_CHARS else ""),
            "source":        source,  # full source for tool execution
        })

    return {
        "title":             parsed_pr.pr_title or "",
        "description":       parsed_pr.pr_description or "",
        "repo":              parsed_pr.repo_full_name or "",
        "changed_files":     parsed_pr.changed_files or [],
        "changed_functions": changed_functions,
    }


def _build_function_sources(pr_context: dict) -> dict:
    """
    Build {function_name: {source, file, start_line}} for tool execution.
    """
    return {
        fn["name"]: {
            "source":     fn["source"],
            "file":       fn["file"],
            "start_line": fn.get("start_line", 1),
        }
        for fn in pr_context["changed_functions"]
        if fn.get("source")
    }


def _validate_tool_plan(tool_plan: list, valid_function_names: set) -> list:
    """
    Validate and clean the tool plan from Gemini's reasoning step.
    Removes unknown tools and functions not in the changed set.
    """
    validated = []
    for entry in tool_plan:
        tool = entry.get("tool", "")
        if tool not in VALID_TOOLS:
            logger.warning(f"Agent requested unknown tool '{tool}', skipping")
            continue

        # Filter functions to only those we actually have source for
        requested_fns = entry.get("functions", [])
        valid_fns = [f for f in requested_fns if f in valid_function_names]

        # If agent listed wrong names, try all available functions as fallback
        if not valid_fns and requested_fns:
            logger.warning(
                f"Tool {tool}: none of {requested_fns} found in changed functions "
                f"(available: {valid_function_names}). Using all available."
            )
            valid_fns = list(valid_function_names)

        if valid_fns:
            validated.append({
                "tool":      tool,
                "functions": valid_fns,
                "reason":    entry.get("reason", ""),
            })

    return validated


# ─── Main Agent Entry Point ───────────────────────────────────────────────────

def run_agent_review(
    parsed_pr,
    complexity_results: list = None,
    repo_path: str = None,
) -> AgentReview:
    """
    Run the full ReAct agent review on a parsed PR.

    Args:
        parsed_pr: ParsedPR object from tools/parse_diff.py
        complexity_results: list of ComplexityResult from tools/complexity_analyzer.py
        repo_path: path to cloned repo (for call graph + semantic search)

    Returns:
        AgentReview with structured verdict + inline comments
    """
    start = time.time()
    review = AgentReview(
        overall_verdict="COMMENT",
        risk_level="LOW",
        summary="",
    )

    # ── Guard: no functions changed ────────────────────────────────────────────
    if not parsed_pr.changed_functions:
        review.summary = (
            "This PR contains no Python function changes. "
            "It may be documentation, config, or non-Python file changes."
        )
        review.overall_verdict = "COMMENT"
        review.confidence = 0.5
        review.elapsed_seconds = time.time() - start
        return review

    try:
        client = _get_gemini_client()
    except ValueError as e:
        review.error = str(e)
        review.summary = f"Agent unavailable: {e}"
        review.elapsed_seconds = time.time() - start
        return review

    try:
        # ── Build context ──────────────────────────────────────────────────────
        pr_context = _build_pr_context(parsed_pr, complexity_results)
        function_sources = _build_function_sources(pr_context)
        valid_fn_names = set(function_sources.keys())

        logger.info(
            f"Agent starting | pr=#{parsed_pr.pr_number} "
            f"functions={len(valid_fn_names)} repo={'yes' if repo_path else 'no'}"
        )

        # ══ CALL 1: REASON ════════════════════════════════════════════════════
        reasoning_prompt = build_reasoning_prompt(pr_context)
        reasoning_text = _call_gemini_with_retry(client, reasoning_prompt, "REASON")
        review.gemini_calls += 1

        reasoning_data = _parse_json_response(reasoning_text, "REASON")
        review.pr_type  = reasoning_data.get("pr_type", "unknown")
        review.reasoning = reasoning_data.get("reasoning", "")
        raw_tool_plan   = reasoning_data.get("tools_to_call", [])
        review.tool_plan = _validate_tool_plan(raw_tool_plan, valid_fn_names)

        logger.info(
            f"Reasoning complete | pr_type={review.pr_type} "
            f"risk={reasoning_data.get('risk_level_initial')} "
            f"tools_planned={[t['tool'] for t in review.tool_plan]}"
        )

        # Rate limit sleep before next work (tools are local, no sleep needed between them)
        time.sleep(RATE_LIMIT_SLEEP)

        # ══ EXECUTE TOOLS (local, no API calls) ═══════════════════════════════
        raw_results = {}
        if review.tool_plan:
            raw_results = execute_tools_for_plan(
                tools_plan=review.tool_plan,
                function_sources=function_sources,
                repo_path=repo_path,
            )
        review.raw_tool_results = raw_results
        review.tools_used = sorted({entry["tool"].lower() for entry in review.tool_plan})

        # Always include complexity in the tool results summary
        complexity_summary = "\n".join(
            f"  {r.function_name}: CC={r.cyclomatic_complexity} "
            f"nesting={r.max_nesting_depth} lines={r.line_count} "
            f"{'⚠ needs refactoring' if r.needs_refactoring else 'OK'}"
            for r in (complexity_results or [])
        )
        formatted_results = format_results_for_prompt(raw_results)
        if complexity_summary:
            formatted_results["get_function_complexity"] = complexity_summary

        # ══ CALL 2: SYNTHESIZE ════════════════════════════════════════════════
        synthesis_prompt = build_synthesis_prompt(pr_context, formatted_results)
        synthesis_text = _call_gemini_with_retry(client, synthesis_prompt, "SYNTHESIZE")
        review.gemini_calls += 1

        synthesis_data = _parse_json_response(synthesis_text, "SYNTHESIZE")

        # ── Map synthesis output to AgentReview ────────────────────────────────
        review.overall_verdict = synthesis_data.get("overall_verdict", "COMMENT")
        review.risk_level       = synthesis_data.get("risk_level", "LOW")
        review.summary          = synthesis_data.get("summary", "")
        review.confidence       = float(synthesis_data.get("confidence", 0.7))

        # Parse inline comments
        for raw_comment in synthesis_data.get("inline_comments", []):
            try:
                review.inline_comments.append(InlineComment(
                    file=raw_comment.get("file", ""),
                    line=int(raw_comment.get("line", 1)),
                    severity=raw_comment.get("severity", "LOW"),
                    category=raw_comment.get("category", "STYLE"),
                    comment=raw_comment.get("comment", ""),
                    suggestion=raw_comment.get("suggestion", ""),
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed inline comment: {raw_comment} — {e}")

        # Merge tools_used from synthesis (Gemini may correct our list)
        synthesis_tools = synthesis_data.get("tools_used", [])
        if synthesis_tools:
            # Normalise to lowercase + deduplicate — Gemini sometimes returns UPPER_CASE
            all_tools = {t.lower() for t in review.tools_used} | {t.lower() for t in synthesis_tools}
            review.tools_used = sorted(all_tools)

        review.elapsed_seconds = time.time() - start
        logger.info(
            f"Agent complete | pr=#{parsed_pr.pr_number} "
            f"verdict={review.overall_verdict} risk={review.risk_level} "
            f"comments={len(review.inline_comments)} "
            f"gemini_calls={review.gemini_calls} "
            f"elapsed={review.elapsed_seconds:.1f}s"
        )
        return review

    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        review.error = str(e)
        review.summary = f"Agent encountered an error: {e}. Check server logs."
        review.overall_verdict = "COMMENT"
        review.elapsed_seconds = time.time() - start
        return review


def agent_review_to_dict(review: AgentReview) -> dict:
    """Serialize AgentReview to a JSON-serializable dict for API responses."""
    return {
        "overall_verdict":  review.overall_verdict,
        "risk_level":       review.risk_level,
        "summary":          review.summary,
        "inline_comments": [
            {
                "file":       c.file,
                "line":       c.line,
                "severity":   c.severity,
                "category":   c.category,
                "comment":    c.comment,
                "suggestion": c.suggestion,
            }
            for c in review.inline_comments
        ],
        "tools_used":       review.tools_used,
        "confidence":       review.confidence,
        "agent_meta": {
            "pr_type":        review.pr_type,
            "reasoning":      review.reasoning,
            "gemini_calls":   review.gemini_calls,
            "elapsed_seconds": round(review.elapsed_seconds, 2),
            "error":          review.error,
        },
    }
