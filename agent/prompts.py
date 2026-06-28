"""
prompts.py — All prompts and output schemas for the ReAct agent.

Design principle: Two Gemini calls per PR, never more.
  Call 1 (REASON):    Given PR context, decide which tools to run and why.
  Call 2 (SYNTHESIZE): Given all tool results, produce the final review.

Batching reasoning + tool selection into ONE call is what keeps us within
free tier limits. We never ask Gemini "what should I do next?" in a loop.
"""

# ─── Tool descriptions fed to the agent ──────────────────────────────────────
# Each description tells the agent WHEN to use the tool, not just what it does.

TOOL_DESCRIPTIONS = {
    "scan_for_vulnerabilities": (
        "FAISS similarity search against 21 CWE-tagged vulnerability patterns "
        "(CWE-89 SQL injection, CWE-78 command injection, CWE-22 path traversal, "
        "CWE-798 hardcoded credentials, CWE-502 insecure deserialization, "
        "CWE-328 weak crypto, etc.). "
        "USE WHEN: function touches databases, user input, files, subprocesses, "
        "auth, crypto, or serialization. Returns CWE ID + fix suggestion."
    ),
    "check_vulnerability_patterns": (
        "Binary CodeBERT classifier fine-tuned on BigVul (188k real CVEs). "
        "Returns VULNERABLE/SAFE label with confidence score. "
        "USE WHEN: scan_for_vulnerabilities flags something OR the function name "
        "suggests security sensitivity (login, auth, execute, parse, deserialize). "
        "Complements FAISS — catches novel patterns the pattern library missed."
    ),
    "build_call_graph": (
        "Builds a NetworkX call graph across the entire repo. Returns fan-in "
        "(how many functions call this one), fan-out, and impact_score (0-1). "
        "USE WHEN: the changed function is called by many others (high blast radius), "
        "or when it's in a shared utility module. Skip for isolated helper functions."
    ),
    "semantic_search": (
        "TF-IDF similarity search to find structurally similar functions in the repo. "
        "USE WHEN: the PR introduces a new pattern — check if a similar pattern "
        "exists elsewhere (catches inconsistencies like 'this function validates input "
        "but the similar one in auth/ does not'). Skip for one-off utility functions."
    ),
    "get_function_complexity": (
        "Cyclomatic complexity + nesting depth. Already computed for all changed "
        "functions — available for free, no need to select. Always included."
    ),
}

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert code reviewer AI embedded in an automated PR review system.
You have access to several analysis tools that run locally (fast, free).
Your job is to reason about a Pull Request and decide which tools will give the most useful signal.

IMPORTANT RULES:
1. Be selective. A simple refactoring PR (renaming, extracting methods, adding comments) needs 1-2 tools max.
   A PR touching auth, databases, file I/O, or serialization needs all security tools.
2. Every tool call must be justified — explain WHY you're calling it for THIS PR.
3. The inline comments you write must be specific and actionable, tied to exact line numbers.
4. Severity levels: CRITICAL (exploitable security bug), HIGH (security concern or significant bug),
   MEDIUM (correctness issue or poor practice), LOW (style or minor suggestion).
5. Categories: SECURITY, CORRECTNESS, STYLE, COVERAGE.
6. overall_verdict: REQUEST_CHANGES if any HIGH/CRITICAL issue found, APPROVE if clean, COMMENT otherwise.
7. confidence reflects how certain you are given the available evidence (0.0-1.0).
"""

# ─── Call 1: Reasoning prompt ─────────────────────────────────────────────────

def build_reasoning_prompt(pr_context: dict) -> str:
    """
    Builds the prompt for the first Gemini call.
    pr_context keys: title, description, repo, changed_files,
                     changed_functions (list of {name, file, complexity, lines_changed, source_preview})
    """
    files_str = ", ".join(pr_context.get("changed_files", []))
    funcs_str = "\n".join(
        f"  - {f['name']} in {f['file']} "
        f"(complexity={f.get('complexity', '?')}, "
        f"changed_lines={f.get('lines_changed', '?')})\n"
        f"    Preview: {f.get('source_preview', '')[:200]}"
        for f in pr_context.get("changed_functions", [])
    )

    tools_str = "\n".join(
        f"  {name}: {desc}"
        for name, desc in TOOL_DESCRIPTIONS.items()
        if name != "get_function_complexity"  # always run, no need to select
    )

    return f"""You are reviewing a GitHub Pull Request. Analyze it and decide which analysis tools to run.

PR INFORMATION:
  Title: {pr_context.get('title', 'N/A')}
  Description: {pr_context.get('description', 'No description provided')}
  Repository: {pr_context.get('repo', 'N/A')}
  Changed files: {files_str}

CHANGED FUNCTIONS:
{funcs_str if funcs_str else '  No function-level changes detected (possibly non-Python files or comment-only changes)'}

AVAILABLE TOOLS (get_function_complexity always runs — select additional ones):
{tools_str}

Respond with a JSON object with exactly these fields:
{{
  "pr_type": "one of: security_fix | feature | refactoring | bug_fix | style | test | documentation | mixed",
  "risk_level_initial": "one of: LOW | MEDIUM | HIGH",
  "reasoning": "2-3 sentences explaining what this PR does and what concerns you have",
  "security_relevant": true or false,
  "tools_to_call": [
    {{"tool": "tool_name", "functions": ["func1", "func2"], "reason": "why for these specific functions"}}
  ]
}}

If no additional tools are needed (e.g. pure documentation or style change), set tools_to_call to [].
Select each tool AT MOST ONCE, listing all relevant functions for it together.
"""

# ─── Call 2: Synthesis prompt ──────────────────────────────────────────────────

def build_synthesis_prompt(pr_context: dict, tool_results: dict) -> str:
    """
    Builds the prompt for the second Gemini call.
    tool_results: dict mapping tool_name -> result summary string
    """
    results_str = "\n\n".join(
        f"=== {tool_name.upper()} ===\n{result}"
        for tool_name, result in tool_results.items()
    )

    funcs_str = "\n".join(
        f"  - {f['name']} (file: {f['file']}, line {f.get('start_line', '?')})"
        for f in pr_context.get("changed_functions", [])
    )

    return f"""You have analyzed a Pull Request using automated tools. Now write the final code review.

PR: {pr_context.get('title')} in {pr_context.get('repo')}

CHANGED FUNCTIONS:
{funcs_str}

TOOL RESULTS:
{results_str if results_str else 'No additional tools were run (PR deemed low-risk).'}

Write a structured code review as a JSON object:
{{
  "overall_verdict": "REQUEST_CHANGES | APPROVE | COMMENT",
  "risk_level": "LOW | MEDIUM | HIGH",
  "summary": "2-3 sentence plain English summary of the PR and main concerns. Be specific.",
  "inline_comments": [
    {{
      "file": "filename.py",
      "line": 42,
      "severity": "CRITICAL | HIGH | MEDIUM | LOW",
      "category": "SECURITY | CORRECTNESS | STYLE | COVERAGE",
      "comment": "specific actionable comment referencing the actual code",
      "suggestion": "concrete fix or alternative code snippet"
    }}
  ],
  "tools_used": ["list of tool names that ran"],
  "confidence": 0.85
}}

Rules:
- inline_comments must reference actual line numbers from the changed functions
- If a vulnerability was flagged, ALWAYS include an inline comment for it
- If no issues found, inline_comments can be [] and verdict is APPROVE
- suggestion should be specific code, not vague advice
- confidence: 0.9+ if multiple tools agree, 0.6-0.8 if uncertain, 0.5 if no security tools ran
"""
