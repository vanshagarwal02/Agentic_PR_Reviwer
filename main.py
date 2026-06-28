"""
main.py — FastAPI webhook server

This is the entry point of the whole system.

HOW THE FLOW WORKS (end to end):
  1. You push code to GitHub, open a PR
  2. GitHub sees there's a webhook configured for this repo
  3. GitHub sends a POST request to your server (your_ngrok_url/webhook)
  4. This server receives it, verifies it's really from GitHub,
     and queues the PR for review
  5. (Week 5) The agent processes it and posts the review back to GitHub

Weeks 1–2, we:
  - Receive the webhook
  - Verify it's authentic
  - Call parse_diff() to get changed functions
  - Call get_function_complexity() on each changed function
  - Call build_call_graph() to measure blast radius (who calls what)
  - Call semantic_search() to find structurally similar functions
  - Return the structured result as JSON

WHY FASTAPI over Flask:
  - Built-in async support (important for Week 5 when we call Gemini)
  - Automatic OpenAPI docs at /docs (great for debugging)
  - Type hints + Pydantic validation built in
  - Faster than Flask for I/O-bound work
"""

import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import config
from tools.parse_diff import ParsedPR, parse_diff, print_parsed_pr
from tools.complexity_analyzer import get_function_complexity
from tools.call_graph import build_call_graph
from tools.semantic_search import semantic_search
from tools.vulnerability_scanner import scan_for_vulnerabilities, ensure_index_exists
from tools.vulnerability_classifier import classify_functions, check_vulnerability_patterns
from agent.react_agent import run_agent_review, agent_review_to_dict
from tools.github_poster import post_review_to_github
from db.database import init_db, save_review, get_reviews, get_review_by_id, get_review_stats, get_risk_trend, get_total_count, set_review_feedback

# ─── Logging Setup ────────────────────────────────────────────────────────────
# Structured logging is essential — Week 5's agent will log every decision here
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_reviewer")


# ─── FastAPI App ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code here runs ONCE when the server starts.
    Use it for one-time setup: loading ML models, DB connections, etc.
    (Week 3: we'll load the FAISS index here)
    (Week 4: we'll load the vulnerability classifier here)
    """
    logger.info("🚀 AI Code Reviewer starting up...")
    logger.info(f"GitHub token configured: {'✓' if config.GITHUB_TOKEN else '✗ MISSING'}")
    logger.info(f"Webhook secret configured: {'✓' if config.GITHUB_WEBHOOK_SECRET else '✗ MISSING'}")
    # Pre-load the CodeBERT model + FAISS index so the first review isn't slow
    try:
        ensure_index_exists()
        logger.info("Vulnerability index loaded ✓")
    except Exception as e:
        logger.warning(f"Vulnerability index failed to load (non-fatal): {e}")
    # Week 7: initialise the reviews database
    try:
        init_db()
        logger.info("Reviews database initialised ✓")
    except Exception as e:
        logger.warning(f"DB init failed (non-fatal): {e}")
    yield  # Server runs here
    logger.info("Shutting down...")


app = FastAPI(
    title="AI Code Reviewer",
    description="Agentic PR review system — ReAct agent with real ML models",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: Allow the React dashboard (Week 7) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Webhook Signature Verification ───────────────────────────────────────────

def verify_github_signature(payload_bytes: bytes, signature_header: str) -> bool:
    """
    Verify that the webhook request actually came from GitHub.

    HOW IT WORKS:
    GitHub computes HMAC-SHA256(your_secret, payload_body) and sends it
    in the X-Hub-Signature-256 header. We compute the same thing on our side.
    If they match, the request is authentic — only GitHub knows our secret.

    WHY THIS MATTERS:
    Without this check, anyone on the internet could POST to your /webhook
    endpoint and trigger a review (or worse, inject malicious PR data).
    """
    if not config.GITHUB_WEBHOOK_SECRET:
        logger.warning("Webhook secret not configured — skipping signature check")
        return True  # Allow during development without secret; fix before production

    if not signature_header or not signature_header.startswith("sha256="):
        return False

    # Extract the hex digest GitHub sent us
    expected_sig = signature_header[len("sha256="):]

    # Compute our own HMAC
    computed = hmac.new(
        config.GITHUB_WEBHOOK_SECRET.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    # compare_digest prevents timing attacks (don't use ==)
    return hmac.compare_digest(computed, expected_sig)


# ─── Repo Cloning Helper ─────────────────────────────────────────────────────

def _clone_pr_repo(repo_full_name: str, head_sha: str) -> str | None:
    """
    Shallow-clone the PR's repo at the exact head commit into a temp directory.
    Returns the temp directory path, or None if cloning fails.

    WHY CLONE:
      call_graph and semantic_search need to scan the ENTIRE codebase of the
      repo being reviewed — not our own ai_code_reviewer project folder.
      A shallow clone (--depth=1) downloads only the latest snapshot of the
      code, not the full history, so it's fast (seconds, not minutes).

    The caller is responsible for deleting the temp directory after use.
    We embed the token in the URL so git can authenticate to private repos.
    """
    if not config.GITHUB_TOKEN:
        logger.warning("No GITHUB_TOKEN — cannot clone repo for call graph / semantic search")
        return None

    clone_url = f"https://{config.GITHUB_TOKEN}@github.com/{repo_full_name}.git"
    tmp_dir = tempfile.mkdtemp(prefix="ai_reviewer_")

    try:
        # --depth=1 = only fetch the latest commit (no history) → much faster
        # --no-tags = skip tag objects → slightly faster
        subprocess.run(
            ["git", "clone", "--depth=1", "--no-tags", clone_url, tmp_dir],
            check=True,
            capture_output=True,  # suppress git output from appearing in logs
            timeout=60,
        )
        logger.info(f"Repo cloned | repo={repo_full_name} dir={tmp_dir}")
        return tmp_dir
    except subprocess.TimeoutExpired:
        logger.warning(f"Clone timed out for {repo_full_name}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None
    except subprocess.CalledProcessError as e:
        logger.warning(f"Clone failed for {repo_full_name}: {e.stderr.decode()[:200]}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None


# ─── Background Task: Process PR ──────────────────────────────────────────────

def process_pr_review(pr_url: str) -> None:
    """
    Called as a background task when a PR webhook arrives.

    WHY BACKGROUND:
    GitHub expects your webhook endpoint to respond in <10 seconds.
    A full review takes 30-60 seconds. So we:
    1. Receive webhook → immediately respond 200 OK to GitHub
    2. Run the actual review in the background
    (Week 5: the agent loop runs here)
    """
    logger.info(f"Starting review | pr={pr_url}")
    start_time = time.time()

    try:
        # ── Step 1: Parse the PR diff ──────────────────────────────────────────
        parsed = parse_diff(pr_url)
        logger.info(
            f"Diff parsed | pr=#{parsed.pr_number} "
            f"functions={len(parsed.changed_functions)} "
            f"files={len(parsed.changed_files)}"
        )

        # ── Step 2: Run complexity analysis on changed functions ───────────────
        complexity_results = []
        for cf in parsed.changed_functions:
            if cf.full_source:  # skip if we couldn't fetch the file
                result = get_function_complexity(cf.full_source, cf.language)
                complexity_results.append(result)
                if result.needs_refactoring:
                    logger.info(
                        f"Complexity flag | fn={cf.function_name} "
                        f"cc={result.cyclomatic_complexity} "
                        f"depth={result.max_nesting_depth}"
                    )

        # ── Step 3: Clone PR repo, then run call graph + semantic search ────────
        # We clone the repo being reviewed (not our own project folder).
        # After analysis, the temp dir is deleted to keep the machine clean.
        repo_dir = _clone_pr_repo(parsed.repo_full_name, head_sha="HEAD")

        call_graph_result = None
        if parsed.changed_functions and repo_dir:
            func_names = [cf.function_name for cf in parsed.changed_functions]
            try:
                call_graph_result = build_call_graph(repo_dir, func_names)
                for impact in call_graph_result.impacts:
                    if impact.impact_score > 0.5:
                        logger.info(
                            f"High-impact function | fn={impact.function_name} "
                            f"score={impact.impact_score:.2f} "
                            f"callers={impact.fan_in}"
                        )
            except Exception as e:
                logger.warning(f"Call graph failed (non-fatal): {e}")

        # ── Step 4: Semantic search — find similar functions ───────────────────
        semantic_results = []
        if repo_dir:
            for cf in parsed.changed_functions:
                if cf.full_source:
                    try:
                        sem = semantic_search(
                            query_source=cf.full_source,
                            query_name=cf.function_name,
                            query_file=cf.file_path,
                            repo_path=repo_dir,
                            top_k=3,
                        )
                        if sem.similar_functions:
                            logger.info(
                                f"Similar functions found | fn={cf.function_name} "
                                f"matches={len(sem.similar_functions)} "
                                f"mode={sem.mode_used}"
                            )
                        semantic_results.append(sem)
                    except Exception as e:
                        logger.warning(f"Semantic search failed for {cf.function_name} (non-fatal): {e}")

        # Clean up the cloned repo — don't leave temp dirs lying around
        if repo_dir:
            shutil.rmtree(repo_dir, ignore_errors=True)
            logger.info("Temp clone removed")

        # ── Step 5: Vulnerability scan ─────────────────────────────────────────
        vuln_results = []
        for cf in parsed.changed_functions:
            if cf.full_source:
                try:
                    vscan = scan_for_vulnerabilities(
                        function_source=cf.full_source,
                        function_name=cf.function_name,
                        file_path=cf.file_path,
                    )
                    if vscan.matches:
                        logger.info(
                            f"Vulnerability flag | fn={cf.function_name} "
                            f"matches={len(vscan.matches)} "
                            f"top={vscan.matches[0].cwe}"
                        )
                    vuln_results.append(vscan)
                except Exception as e:
                    logger.warning(f"Vuln scan failed for {cf.function_name} (non-fatal): {e}")

        # ── Step 6: Vulnerability classifier (Week 4 — fine-tuned CodeBERT) ────────
        # Runs the binary classifier alongside the Week 3 FAISS scanner.
        # Two independent signals = higher confidence when they agree.
        classifier_results = []
        fn_batch = [
            {"source": cf.full_source, "name": cf.function_name, "file": cf.file_path}
            for cf in parsed.changed_functions
            if cf.full_source
        ]
        if fn_batch:
            try:
                clf_result = classify_functions(fn_batch)
                for pred in clf_result.predictions:
                    if pred.label == "VULNERABLE":
                        logger.info(
                            f"Classifier flag | fn={pred.function_name} "
                            f"confidence={pred.confidence:.3f} "
                            f"method={pred.method}"
                        )
                    classifier_results.append(pred)
            except Exception as e:
                logger.warning(f"Classifier failed (non-fatal): {e}")

        # ── Step 7: Log summary ────────────────────────────────────────────────
        elapsed = time.time() - start_time
        flagged = sum(1 for r in complexity_results if r.needs_refactoring)
        vuln_flagged = sum(1 for v in vuln_results if v.matches)
        clf_flagged = sum(1 for p in classifier_results if p.label == "VULNERABLE")
        logger.info(
            f"Analysis complete | pr=#{parsed.pr_number} "
            f"elapsed={elapsed:.1f}s flagged={flagged} "
            f"faiss_vuln={vuln_flagged} clf_vuln={clf_flagged}"
        )

        _log_review_decision(
            pr_url, parsed, complexity_results,
            call_graph_result, semantic_results,
            vuln_results, classifier_results,
        )

    except Exception as e:
        logger.error(f"[TASK] Review failed for {pr_url}: {e}", exc_info=True)


def _log_review_decision(
    pr_url, parsed, complexity_results,
    call_graph_result=None, semantic_results=None,
    vuln_results=None, classifier_results=None,
):
    """Write a structured log of what we found — essential for debugging."""
    import os
    log_path = os.path.join(config.LOGS_DIR, f"pr_{parsed.pr_number}.json")

    log_data = {
        "pr_url": pr_url,
        "pr_number": parsed.pr_number,
        "pr_title": parsed.pr_title,
        "changed_files": parsed.changed_files,
        "changed_functions": [
            {
                "file": cf.file_path,
                "function": cf.function_name,
                "lines": cf.changed_lines,
                "is_new": cf.is_new_file,
            }
            for cf in parsed.changed_functions
        ],
        "complexity_results": [
            {
                "function": r.function_name,
                "cyclomatic_complexity": r.cyclomatic_complexity,
                "nesting_depth": r.max_nesting_depth,
                "line_count": r.line_count,
                "needs_refactoring": r.needs_refactoring,
                "comment": r.review_comment,
            }
            for r in complexity_results
        ],
        "call_graph": [
            {
                "function": imp.function_name,
                "impact_score": round(imp.impact_score, 3),
                "fan_in": imp.fan_in,
                "fan_out": imp.fan_out,
                "direct_callers": imp.direct_callers,
                "direct_callees": imp.direct_callees,
                "transitive_caller_count": len(imp.all_callers),
                "is_high_impact": imp.impact_score > 0.5,
            }
            for imp in (call_graph_result.impacts if call_graph_result else [])
        ],
        "semantic_similar": [
            {
                "query": sem.query_function_name,
                "mode": sem.mode_used,
                "similar": [
                    {
                        "function": sf.function_name,
                        "file": sf.file_path,
                        "score": sf.similarity_score,
                        "reason": sf.match_reason,
                    }
                    for sf in sem.similar_functions
                ],
            }
            for sem in (semantic_results or [])
        ],
        "vulnerability_scan": [
            {
                "function": v.function_name,
                "file": v.file_path,
                "scanned": v.scanned,
                "matches": [
                    {
                        "cwe": m.cwe,
                        "name": m.name,
                        "severity": m.severity,
                        "score": m.similarity_score,
                        "description": m.description,
                        "fix": m.fix,
                    }
                    for m in v.matches
                ],
            }
            for v in (vuln_results or [])
        ],
        "classifier_predictions": [
            {
                "function": p.function_name,
                "file": p.file_path,
                "label": p.label,
                "confidence": p.confidence,
                "vulnerable_probability": p.vulnerable_probability,
                "method": p.method,
                "fallback_used": p.fallback_used,
            }
            for p in (classifier_results or [])
        ],
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"Review log saved | path={log_path}")


# ─── API Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    Used by: ngrok to verify server is up, monitoring tools.
    Visit http://localhost:8000/health in your browser to confirm server is running.
    """
    return {
        "status": "ok",
        "github_configured": bool(config.GITHUB_TOKEN),
        "webhook_secret_configured": bool(config.GITHUB_WEBHOOK_SECRET),
    }


@app.post("/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Receives GitHub webhook events.

    GitHub sends different event types (push, pull_request, issue, etc.)
    We only care about pull_request events with action "opened" or "synchronize".
    "synchronize" = someone pushed new commits to an existing PR.
    """
    # Read raw body bytes (needed for signature verification)
    payload_bytes = await request.body()

    # ── Verify signature ──────────────────────────────────────────────────────
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not verify_github_signature(payload_bytes, signature):
        logger.warning("Webhook signature verification FAILED — rejecting request")
        raise HTTPException(status_code=403, detail="Invalid signature")

    # ── Parse JSON payload ────────────────────────────────────────────────────
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event_type = request.headers.get("X-GitHub-Event", "unknown")
    logger.info(f"Received webhook: event={event_type}, action={payload.get('action', 'none')}")

    # ── Only process pull_request events ─────────────────────────────────────
    if event_type != "pull_request":
        return Response(status_code=200, content="Event ignored (not a PR event)")

    action = payload.get("action", "")
    if action not in ("opened", "synchronize", "reopened"):
        return Response(status_code=200, content=f"Action '{action}' ignored")

    # ── Extract PR URL ────────────────────────────────────────────────────────
    pr_html_url = payload.get("pull_request", {}).get("html_url")
    if not pr_html_url:
        raise HTTPException(status_code=400, detail="Could not extract PR URL from payload")

    logger.info(f"PR event: action={action}, url={pr_html_url}")

    # ── Queue the agent review as a background task ────────────────────────────
    # We respond to GitHub immediately (within 10s) and process in the background.
    # Results saved to logs/agent_pr_<N>.json
    background_tasks.add_task(_run_agent_background, pr_html_url)

    return {
        "status": "queued",
        "pr_url": pr_html_url,
        "message": "Review queued — check logs for progress",
    }


@app.post("/review")
async def manual_review(body: dict):
    """
    Manual trigger endpoint — useful for testing without a webhook.
    POST {"pr_url": "https://github.com/owner/repo/pull/123"}

    Try it at http://localhost:8000/docs (Swagger UI auto-generated by FastAPI)
    """
    pr_url = body.get("pr_url")
    if not pr_url:
        raise HTTPException(status_code=400, detail="Missing 'pr_url' in request body")

    logger.info(f"Manual review triggered for: {pr_url}")

    # For the manual endpoint, run synchronously so we can return the full result
    try:
        # ── Step 1: Parse diff ────────────────────────────────────────────────
        parsed = parse_diff(pr_url)

        # ── Step 2: Complexity ────────────────────────────────────────────────
        complexity_results = []
        for cf in parsed.changed_functions:
            if cf.full_source:
                result = get_function_complexity(cf.full_source, cf.language)
                complexity_results.append({
                    "function": result.function_name,
                    "file": cf.file_path,
                    "cyclomatic_complexity": result.cyclomatic_complexity,
                    "complexity_label": result.complexity_label,
                    "nesting_depth": result.max_nesting_depth,
                    "line_count": result.line_count,
                    "parameter_count": result.parameter_count,
                    "has_recursion": result.has_recursion,
                    "needs_refactoring": result.needs_refactoring,
                    "review_comment": result.review_comment,
                })

        # ── Step 3: Clone PR repo, then run call graph + semantic search ────────
        repo_dir = _clone_pr_repo(parsed.repo_full_name, head_sha="HEAD")

        call_graph_data = []
        if parsed.changed_functions and repo_dir:
            func_names = [cf.function_name for cf in parsed.changed_functions]
            try:
                cg = build_call_graph(repo_dir, func_names)
                call_graph_data = [
                    {
                        "function": imp.function_name,
                        "impact_score": round(imp.impact_score, 3),
                        "fan_in": imp.fan_in,
                        "fan_out": imp.fan_out,
                        "direct_callers": imp.direct_callers,
                        "direct_callees": imp.direct_callees,
                        "transitive_caller_count": len(imp.all_callers),
                        "is_high_impact": imp.impact_score > 0.5,
                    }
                    for imp in cg.impacts
                ]
            except Exception as e:
                logger.warning(f"Call graph failed (non-fatal): {e}")

        # ── Step 4: Semantic search ───────────────────────────────────────────
        semantic_data = []
        if repo_dir:
            for cf in parsed.changed_functions:
                if cf.full_source:
                    try:
                        sem = semantic_search(
                            query_source=cf.full_source,
                            query_name=cf.function_name,
                            query_file=cf.file_path,
                            repo_path=repo_dir,
                            top_k=3,
                        )
                        semantic_data.append({
                            "query_function": cf.function_name,
                            "mode": sem.mode_used,
                            "searched": sem.total_functions_searched,
                            "similar": [
                                {
                                    "function": sf.function_name,
                                    "file": sf.file_path,
                                    "score": sf.similarity_score,
                                    "reason": sf.match_reason,
                                }
                                for sf in sem.similar_functions
                            ],
                        })
                    except Exception as e:
                        logger.warning(f"Semantic search failed for {cf.function_name} (non-fatal): {e}")

        # Clean up the cloned repo
        if repo_dir:
            shutil.rmtree(repo_dir, ignore_errors=True)

        # ── Step 5: Vulnerability scan ────────────────────────────────────────
        vuln_data = []
        for cf in parsed.changed_functions:
            if cf.full_source:
                try:
                    vscan = scan_for_vulnerabilities(
                        function_source=cf.full_source,
                        function_name=cf.function_name,
                        file_path=cf.file_path,
                    )
                    vuln_data.append({
                        "function": cf.function_name,
                        "file": cf.file_path,
                        "scanned": vscan.scanned,
                        "patterns_checked": vscan.functions_in_index,
                        "matches": [
                            {
                                "cwe": m.cwe,
                                "name": m.name,
                                "severity": m.severity,
                                "similarity_score": m.similarity_score,
                                "description": m.description,
                                "fix": m.fix,
                            }
                            for m in vscan.matches
                        ],
                    })
                except Exception as e:
                    logger.warning(f"Vuln scan failed for {cf.function_name} (non-fatal): {e}")

        # ── Step 6: Vulnerability classifier (Week 4) ─────────────────────────
        clf_data = []
        fn_batch = [
            {"source": cf.full_source, "name": cf.function_name, "file": cf.file_path}
            for cf in parsed.changed_functions if cf.full_source
        ]
        if fn_batch:
            try:
                clf_result = classify_functions(fn_batch)
                clf_data = [
                    {
                        "function": p.function_name,
                        "file": p.file_path,
                        "label": p.label,
                        "confidence": p.confidence,
                        "vulnerable_probability": p.vulnerable_probability,
                        "method": p.method,
                        "fallback_used": p.fallback_used,
                    }
                    for p in clf_result.predictions
                ]
            except Exception as e:
                logger.warning(f"Classifier failed (non-fatal): {e}")

        return {
            "pr_number": parsed.pr_number,
            "pr_title": parsed.pr_title,
            "repo": parsed.repo_full_name,
            "changed_files": parsed.changed_files,
            "changed_functions": [
                {
                    "file": cf.file_path,
                    "function": cf.function_name,
                    "language": cf.language,
                    "start_line": cf.start_line,
                    "end_line": cf.end_line,
                    "changed_lines": cf.changed_lines,
                    "is_new_file": cf.is_new_file,
                }
                for cf in parsed.changed_functions
            ],
            "complexity_analysis": complexity_results,
            "call_graph": call_graph_data,
            "semantic_similar": semantic_data,
            "vulnerability_scan": vuln_data,
            "classifier_predictions": clf_data,
        }

    except Exception as e:
        logger.error(f"Manual review failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Agent Endpoints (Week 5) ─────────────────────────────────────────────────

@app.post("/agent/review/manual", tags=["Agent"])
async def agent_manual_review(request: Request):
    """
    Synchronous agent review. Blocks until complete (~20-40 seconds).
    Returns the full structured review with inline comments.
    """
    body = await request.json()
    pr_url = body.get("pr_url")
    if not pr_url:
        raise HTTPException(status_code=400, detail="Missing 'pr_url' in request body")
    pr_url = str(pr_url)
    logger.info(f"Agent manual review requested | url={pr_url}")

    try:
        # Step 1: Parse the PR
        parsed = parse_diff(pr_url)

        # Step 2: Run complexity (always — agent gets it for free)
        complexity_results = []
        for cf in parsed.changed_functions:
            if cf.full_source:
                try:
                    cr = get_function_complexity(cf.full_source, cf.function_name)
                    complexity_results.append(cr)
                except Exception as e:
                    logger.warning(f"Complexity failed for {cf.function_name}: {e}")

        # Step 3: Clone repo for tools that need it (call graph, semantic search)
        repo_dir = None
        try:
            repo_dir = _clone_pr_repo(parsed.repo_full_name, head_sha="HEAD")
        except Exception as e:
            logger.warning(f"Repo clone failed (call graph/semantic search unavailable): {e}")

        # Step 4: Run the agent
        review = run_agent_review(
            parsed_pr=parsed,
            complexity_results=complexity_results,
            repo_path=repo_dir,
        )

        # Step 5: Cleanup
        if repo_dir:
            shutil.rmtree(repo_dir, ignore_errors=True)

        # Step 6: Log to file
        result = agent_review_to_dict(review)
        log_path = os.path.join(config.LOGS_DIR, f"agent_pr_{parsed.pr_number}.json")
        with open(log_path, "w") as f:
            json.dump({"pr_url": pr_url, "pr_number": parsed.pr_number,
                       "pr_title": parsed.pr_title, **result}, f, indent=2)
        logger.info(f"Agent review log saved | path={log_path}")

        # Step 7: Persist to database (Week 7)
        save_review(
            pr_url=pr_url,
            pr_number=parsed.pr_number,
            pr_title=parsed.pr_title or "",
            repo=parsed.repo_full_name or "",
            review_dict=result,
        )

        return {
            "pr_url":   pr_url,
            "pr_number": parsed.pr_number,
            "pr_title":  parsed.pr_title,
            **result,
        }

    except Exception as e:
        logger.error(f"Agent manual review failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/review", tags=["Agent"])
async def agent_background_review(request: Request, background_tasks: BackgroundTasks):
    """
    Async agent review — returns immediately, runs in background.
    Check logs/agent_pr_<number>.json for results.
    """
    body = await request.json()
    pr_url = body.get("pr_url")
    if not pr_url:
        raise HTTPException(status_code=400, detail="Missing 'pr_url' in request body")
    pr_url = str(pr_url)
    background_tasks.add_task(_run_agent_background, pr_url)
    return {"status": "queued", "pr_url": pr_url,
            "message": "Agent review started. Check logs/agent_pr_<N>.json for results."}


async def _run_agent_background(pr_url: str, auto_post: bool = True):
    """Background task wrapper for agent review + optional GitHub posting."""
    try:
        parsed = parse_diff(pr_url)
        complexity_results = []
        for cf in parsed.changed_functions:
            if cf.full_source:
                try:
                    complexity_results.append(
                        get_function_complexity(cf.full_source, cf.function_name)
                    )
                except Exception:
                    pass

        repo_dir = None
        try:
            repo_dir = _clone_pr_repo(parsed.repo_full_name, head_sha="HEAD")
        except Exception:
            pass

        review = run_agent_review(parsed, complexity_results, repo_dir)

        if repo_dir:
            shutil.rmtree(repo_dir, ignore_errors=True)

        result = agent_review_to_dict(review)
        log_path = os.path.join(config.LOGS_DIR, f"agent_pr_{parsed.pr_number}.json")
        with open(log_path, "w") as f:
            json.dump({"pr_url": pr_url, "pr_number": parsed.pr_number,
                       "pr_title": parsed.pr_title, **result}, f, indent=2)
        logger.info(f"Background agent review complete | pr=#{parsed.pr_number}")

        # Week 7: persist to database
        save_review(
            pr_url=pr_url,
            pr_number=parsed.pr_number,
            pr_title=parsed.pr_title or "",
            repo=parsed.repo_full_name or "",
            review_dict=result,
        )

        # ── Auto-post review to GitHub ─────────────────────────────────────────
        if auto_post and not review.error:
            post_result = post_review_to_github(pr_url, result)
            if post_result.success:
                logger.info(
                    f"GitHub review posted | pr=#{parsed.pr_number} "
                    f"inline={post_result.inline_posted} "
                    f"fallback={post_result.inline_fallback} "
                    f"url={post_result.review_url}"
                )
            else:
                logger.warning(
                    f"GitHub post failed | pr=#{parsed.pr_number} "
                    f"error={post_result.error}"
                )
        elif review.error:
            logger.info(f"Skipping GitHub post — agent had error | pr=#{parsed.pr_number}")

    except Exception as e:
        logger.error(f"Background agent review failed: {e}", exc_info=True)


@app.post("/post-review", tags=["Agent"])
async def manual_post_review(request: Request):
    """
    Manually post an existing agent review to GitHub.
    Reads from logs/agent_pr_<N>.json and posts to GitHub.

    Body: {"pr_url": "https://github.com/owner/repo/pull/123"}
    """
    body = await request.json()
    pr_url = body.get("pr_url")
    if not pr_url:
        raise HTTPException(status_code=400, detail="Missing 'pr_url'")

    # Parse PR number from URL
    m = re.search(r"/pull/(\d+)", pr_url)
    if not m:
        raise HTTPException(status_code=400, detail="Cannot parse PR number from URL")
    pr_number = int(m.group(1))

    log_path = os.path.join(config.LOGS_DIR, f"agent_pr_{pr_number}.json")
    if not os.path.exists(log_path):
        raise HTTPException(
            status_code=404,
            detail=f"No agent review found at {log_path}. Run /agent/review/manual first."
        )

    with open(log_path) as f:
        review_data = json.load(f)

    post_result = post_review_to_github(pr_url, review_data)

    if post_result.success:
        return {
            "success": True,
            "review_url":      post_result.review_url,
            "inline_posted":   post_result.inline_posted,
            "inline_fallback": post_result.inline_fallback,
        }
    else:
        raise HTTPException(status_code=500, detail=post_result.error)


# ─── Dashboard API Endpoints (Week 7) ────────────────────────────────────────

@app.get("/api/reviews", tags=["Dashboard"])
async def api_get_reviews(
    repo: str = None,
    verdict: str = None,
    risk_level: str = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Return a paginated list of past reviews for the dashboard table.

    Query params:
      repo       — filter by "owner/repo" (optional)
      verdict    — filter by REQUEST_CHANGES | APPROVE | COMMENT (optional)
      risk_level — filter by LOW | MEDIUM | HIGH (optional)
      limit      — page size (default 50, max 100)
      offset     — pagination offset (default 0)

    Returns:
      { total, reviews: [...] }
    """
    rows = get_reviews(repo=repo, verdict=verdict, risk_level=risk_level,
                       limit=limit, offset=offset)
    total = get_total_count(repo=repo)
    return {"total": total, "reviews": rows}


@app.get("/api/reviews/stats", tags=["Dashboard"])
async def api_get_stats(repo: str = None):
    """
    Return aggregate statistics for the dashboard header cards.

    Returns:
    {
      "total": 42,
      "by_verdict": {"REQUEST_CHANGES": 20, "APPROVE": 18, "COMMENT": 4},
      "by_risk":    {"HIGH": 10, "MEDIUM": 15, "LOW": 17},
      "avg_confidence": 0.84,
      "repos": ["owner/repo1", "owner/repo2"]
    }
    """
    return get_review_stats(repo=repo)


@app.get("/api/reviews/trend", tags=["Dashboard"])
async def api_get_trend(repo: str = None, days: int = 30):
    """
    Return daily risk-level counts for the trend line chart.

    Returns a list of {date, LOW, MEDIUM, HIGH} dicts covering the last `days` days.
    Missing days (no reviews) are filled with zeros.
    """
    days = max(7, min(days, 365))  # clamp between 7 and 365
    return get_risk_trend(repo=repo, days=days)


@app.get("/api/reviews/{review_id}", tags=["Dashboard"])
async def api_get_review(review_id: int):
    """
    Return the full detail for a single review, including inline comments
    and the complete raw agent output.

    Returns 404 if the review id doesn't exist.
    """
    row = get_review_by_id(review_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    return row


# ─── Feedback Endpoint (Week 8) ────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    score: int  # 1 = thumbs up, -1 = thumbs down, 0 = clear


@app.post("/api/reviews/{review_id}/feedback", tags=["Dashboard"])
async def api_post_feedback(review_id: int, body: FeedbackRequest):
    """
    Store developer feedback (thumbs up / down) for a review.

    Body: { "score": 1 }   → thumbs up
          { "score": -1 }  → thumbs down
          { "score": 0 }   → clear / remove feedback

    Returns 404 if the review doesn't exist.
    Returns 400 if score is not in {-1, 0, 1}.
    """
    if body.score not in (1, -1, 0):
        raise HTTPException(status_code=400, detail="score must be 1, -1, or 0")
    ok = set_review_feedback(review_id, body.score)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    return {"success": True, "review_id": review_id, "feedback": body.score}


@app.get("/api/eval/results", tags=["Dashboard"])
async def api_get_eval_results():
    """
    Return the latest evaluation results from data/eval_results.json.
    Returns 404 if evaluate.py has not been run yet.
    """
    results_path = os.path.join(config.BASE_DIR, "data", "eval_results.json")
    if not os.path.exists(results_path):
        raise HTTPException(
            status_code=404,
            detail="No evaluation results found. Run: python scripts/evaluate.py"
        )
    with open(results_path) as f:
        return json.load(f)



if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",   # Listen on all interfaces (needed for ngrok tunneling)
        port=8000,
        reload=True,       # Auto-reload when you save a file (dev mode)
        log_level="info",
    )
