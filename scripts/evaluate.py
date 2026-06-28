#!/usr/bin/env python3
"""
scripts/evaluate.py — Week 8 Evaluation Framework
===================================================

Benchmarks the AI Code Reviewer agent against a curated dataset of
known-vulnerable and known-safe Python functions.

WHY EVALUATION MATTERS:
  Weeks 1–7 built the system. Week 8 proves it works.
  Without evaluation, you can't answer "what's your false-positive rate?"
  or "does it catch SQL injections reliably?" in an interview.

HOW IT WORKS:
  1. Load eval_benchmark.json (20 hand-labeled code samples).
  2. For each sample, construct a minimal ParsedPR object and invoke
     run_agent_review() — the same agent used in production.
  3. Compare agent output (verdict, risk_level) to ground truth labels.
  4. Compute precision, recall, F1, per-category recall.
  5. Save detailed results to data/eval_results.json.

USAGE:
  cd /path/to/ai_code_reviewer
  source venv/bin/activate
  python scripts/evaluate.py                          # uses default benchmark
  python scripts/evaluate.py --dataset data/my.json  # custom dataset
  python scripts/evaluate.py --dry-run               # validate dataset only

COST:
  2 Gemini calls per sample × 20 samples = 40 calls (~3-4 minutes).
  The free Gemini tier allows 15 req/min — the agent's built-in rate-limit
  sleep(4) keeps us safely under that.

OUTPUT:
  Prints a report table to stdout.
  Saves data/eval_results.json for dashboard display.
"""

import argparse
import json
import os
import sys

# ── Quick sanity check: are we inside the venv? ──────────────────────────────
# google-genai and other project deps are only installed in venv/lib, not system
# Python. This check catches the common mistake of running the script without
# first running `source venv/bin/activate`.
import importlib.util as _ilu
if _ilu.find_spec("google.genai") is None and "--dry-run" not in sys.argv:
    print(
        "\n❌  google.genai not found — looks like the virtualenv isn't active.\n"
        "    Run:\n\n"
        "        source venv/bin/activate\n"
        "        python3 scripts/evaluate.py\n",
        file=sys.stderr,
    )
    sys.exit(1)

import time
from datetime import datetime
from typing import Optional

# Add project root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


# ─── Build minimal ParsedPR objects for each benchmark sample ─────────────────
# We import the REAL dataclasses from parse_diff.py so run_agent_review receives
# objects with exactly the right fields — no attribute surprises.

from tools.parse_diff import ParsedPR, ChangedFunction


def _build_fake_pr(sample: dict, pr_number: int) -> ParsedPR:
    """
    Construct a real ParsedPR from a benchmark sample.

    Only one ChangedFunction is created per sample (the function under test).
    The diff_text is synthesised as all-added lines so the agent treats the
    whole function body as new code.
    """
    source = sample["function_source"]
    n_lines = source.count("\n") + 1

    changed_fn = ChangedFunction(
        file_path=sample["file_path"],
        function_name=sample["function_name"],
        language="python",
        start_line=1,
        end_line=n_lines,
        changed_lines=list(range(1, n_lines + 1)),
        full_source=source,
        context_before="",
        context_after="",
        is_new_file=True,
    )

    return ParsedPR(
        pr_number=pr_number,
        pr_title=f"[EVAL] {sample['id']}",
        pr_description=sample.get("description", ""),
        repo_full_name="eval/benchmark",
        base_branch="main",
        head_branch="eval",
        changed_functions=[changed_fn],
        changed_files=[sample["file_path"]],
        total_additions=n_lines,
        total_deletions=0,
    )


# ─── Metric helpers ───────────────────────────────────────────────────────────

def _is_flagged(verdict: str) -> bool:
    """An agent verdict counts as 'flagged' if it's REQUEST_CHANGES."""
    return verdict == "REQUEST_CHANGES"


def _compute_metrics(results: list[dict]) -> dict:
    """
    Compute precision, recall, F1 from result rows.

    Each row has:
        expected_verdict: str
        actual_verdict:   str
        expected_risk:    str
        actual_risk:      str
        vulnerability_type: str
    """
    truly_vulnerable = [r for r in results if _is_flagged(r["expected_verdict"])]
    truly_safe       = [r for r in results if not _is_flagged(r["expected_verdict"])]

    # True positives: vulnerable AND flagged by agent
    tp = [r for r in truly_vulnerable if _is_flagged(r["actual_verdict"])]
    # False negatives: vulnerable BUT NOT flagged
    fn = [r for r in truly_vulnerable if not _is_flagged(r["actual_verdict"])]
    # False positives: safe BUT flagged
    fp = [r for r in truly_safe if _is_flagged(r["actual_verdict"])]
    # True negatives: safe AND not flagged
    tn = [r for r in truly_safe if not _is_flagged(r["actual_verdict"])]

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 1.0
    recall    = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 1.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # Risk level accuracy (among vulnerable samples only)
    risk_correct = sum(
        1 for r in truly_vulnerable
        if r["actual_risk"] == r["expected_risk"]
    )
    risk_accuracy = risk_correct / len(truly_vulnerable) if truly_vulnerable else 0.0

    # Per-category recall
    categories = {}
    for r in truly_vulnerable:
        cat = r["vulnerability_type"]
        categories.setdefault(cat, {"total": 0, "caught": 0})
        categories[cat]["total"] += 1
        if _is_flagged(r["actual_verdict"]):
            categories[cat]["caught"] += 1
    per_cat = {
        cat: {
            "recall": v["caught"] / v["total"],
            "caught": v["caught"],
            "total":  v["total"],
        }
        for cat, v in categories.items()
    }

    return {
        "total":          len(results),
        "vulnerable":     len(truly_vulnerable),
        "safe":           len(truly_safe),
        "tp":             len(tp),
        "fp":             len(fp),
        "fn":             len(fn),
        "tn":             len(tn),
        "precision":      round(precision, 3),
        "recall":         round(recall, 3),
        "f1":             round(f1, 3),
        "risk_accuracy":  round(risk_accuracy, 3),
        "per_category":   per_cat,
    }


# ─── Pretty-print report ──────────────────────────────────────────────────────

def _print_report(metrics: dict, results: list[dict], elapsed: float, gemini_calls: int):
    W = 58
    sep = "─" * W

    print(f"\n{'═' * W}")
    print(f"  AI Code Reviewer — Evaluation Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * W}")
    print(f"\n  Dataset : {metrics['total']} samples "
          f"({metrics['vulnerable']} vulnerable, {metrics['safe']} safe)")
    print(f"  Elapsed : {elapsed:.1f}s   Gemini calls: {gemini_calls}\n")
    print(sep)
    print(f"  {'Metric':<22} {'Value':>10}   {'Raw':>12}")
    print(sep)
    print(f"  {'Precision':<22} {metrics['precision']:>10.1%}   "
          f"  {metrics['tp']} TP / {metrics['tp']+metrics['fp']} flagged")
    print(f"  {'Recall':<22} {metrics['recall']:>10.1%}   "
          f"  {metrics['tp']} caught / {metrics['vulnerable']} vuln")
    print(f"  {'F1 Score':<22} {metrics['f1']:>10.1%}")
    print(f"  {'Risk level accuracy':<22} {metrics['risk_accuracy']:>10.1%}   "
          f"  {round(metrics['risk_accuracy']*metrics['vulnerable'])}"
          f" / {metrics['vulnerable']} correct")
    print(sep)

    # Per-category
    if metrics["per_category"]:
        print(f"\n  Per-category recall:")
        for cat, v in sorted(metrics["per_category"].items()):
            bar_len = int(v["recall"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {cat:<28} {bar}  {v['recall']:>5.0%}  ({v['caught']}/{v['total']})")

    # False positives
    fp_rows = [r for r in results
               if not _is_flagged(r["expected_verdict"]) and _is_flagged(r["actual_verdict"])]
    fn_rows = [r for r in results
               if _is_flagged(r["expected_verdict"]) and not _is_flagged(r["actual_verdict"])]

    if fp_rows:
        print(f"\n  ⚠ False positives ({len(fp_rows)}) — safe code flagged as risky:")
        for r in fp_rows:
            print(f"    · {r['id']}  →  agent said {r['actual_verdict']} / {r['actual_risk']}")
    if fn_rows:
        print(f"\n  ✗ False negatives ({len(fn_rows)}) — missed vulnerabilities:")
        for r in fn_rows:
            print(f"    · {r['id']} ({r['vulnerability_type']})  →  agent said {r['actual_verdict']}")

    if not fp_rows and not fn_rows:
        print(f"\n  ✓ Perfect score! No false positives or false negatives.")

    print(f"\n{'═' * W}\n")


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(dataset_path: str, dry_run: bool = False) -> dict:
    """
    Run the full evaluation pipeline.

    Args:
        dataset_path: Path to eval_benchmark.json
        dry_run:      If True, validate dataset only (no agent calls)

    Returns:
        dict with "metrics" and "results" keys.
    """
    # Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples from {dataset_path}")

    # Validate required fields
    required = {"id", "function_name", "file_path", "function_source",
                "expected_verdict", "expected_risk", "vulnerability_type"}
    for sample in dataset:
        missing = required - set(sample.keys())
        if missing:
            raise ValueError(f"Sample {sample.get('id', '?')} missing fields: {missing}")

    if dry_run:
        print("Dry run — dataset is valid. Exiting without agent calls.")
        return {"metrics": {}, "results": [], "dry_run": True}

    # Import agent (after path is set up)
    from agent.react_agent import run_agent_review, agent_review_to_dict

    results = []
    total_gemini_calls = 0
    start_time = time.time()

    for i, sample in enumerate(dataset):
        sample_id = sample["id"]
        print(f"[{i+1:02d}/{len(dataset)}] {sample_id} ...", end=" ", flush=True)

        fake_pr = _build_fake_pr(sample, pr_number=1000 + i)

        try:
            review = run_agent_review(parsed_pr=fake_pr, complexity_results={}, repo_path=None)
            review_dict = agent_review_to_dict(review)

            actual_verdict = review_dict.get("overall_verdict", "COMMENT")
            actual_risk    = review_dict.get("risk_level", "LOW")
            gemini_calls   = review_dict.get("agent_meta", {}).get("gemini_calls", 2)
            total_gemini_calls += gemini_calls

            result_row = {
                "id":                sample_id,
                "expected_verdict":  sample["expected_verdict"],
                "expected_risk":     sample["expected_risk"],
                "actual_verdict":    actual_verdict,
                "actual_risk":       actual_risk,
                "vulnerability_type": sample["vulnerability_type"],
                "cwe":               sample.get("cwe"),
                "description":       sample.get("description", ""),
                "summary":           review_dict.get("summary", ""),
                "confidence":        review_dict.get("confidence", 0.0),
                "tools_used":        review_dict.get("tools_used", []),
                "gemini_calls":      gemini_calls,
            }

            # Outcome label for readability
            exp_flagged = _is_flagged(sample["expected_verdict"])
            act_flagged = _is_flagged(actual_verdict)
            if exp_flagged and act_flagged:
                outcome = "TP"
            elif exp_flagged and not act_flagged:
                outcome = "FN"
            elif not exp_flagged and act_flagged:
                outcome = "FP"
            else:
                outcome = "TN"
            result_row["outcome"] = outcome
            results.append(result_row)

            print(f"{outcome}  (verdict={actual_verdict}, risk={actual_risk})")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "id":               sample_id,
                "expected_verdict": sample["expected_verdict"],
                "expected_risk":    sample["expected_risk"],
                "actual_verdict":   "ERROR",
                "actual_risk":      "UNKNOWN",
                "vulnerability_type": sample["vulnerability_type"],
                "outcome":          "ERR",
                "error":            str(e),
            })

    elapsed = time.time() - start_time
    metrics = _compute_metrics([r for r in results if r.get("outcome") != "ERR"])
    _print_report(metrics, results, elapsed, total_gemini_calls)

    # Save results
    output = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_path":  dataset_path,
        "elapsed_seconds": round(elapsed, 1),
        "gemini_calls":  total_gemini_calls,
        "metrics":       metrics,
        "results":       results,
    }
    results_path = os.path.join(_ROOT, "data", "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {results_path}")

    return output


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the AI Code Reviewer agent on a labeled benchmark."
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(_ROOT, "data", "eval_benchmark.json"),
        help="Path to the benchmark JSON file (default: data/eval_benchmark.json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset structure without making any agent/Gemini calls"
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: dataset not found at {args.dataset}", file=sys.stderr)
        sys.exit(1)

    run_evaluation(args.dataset, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
