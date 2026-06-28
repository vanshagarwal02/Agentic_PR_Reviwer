#!/usr/bin/env python3
"""
scripts/real_eval.py — Real-World CVE Evaluation
=================================================

Validates the AI Code Reviewer against REAL historical vulnerabilities from
the CVEfixes dataset (Zenodo / HuggingFace) — functions where a CVE was
later filed and the patch is documented.

This turns synthetic benchmark numbers into a defensible, citable claim:
  "Validated on N real CVE-patched Python functions (CWEs 89/78/22/502/798)
   — X% recall, Y% precision on production-level vulnerabilities."

HOW IT WORKS:
  1. Downloads CVEfixes from HuggingFace (uses `datasets` lib already in venv).
  2. Filters to Python functions matching your 5 CWE categories.
  3. Samples up to --n-vulnerable vulnerable + equal safe counterparts.
  4. Runs through the SAME agent pipeline as production (run_agent_review).
  5. Computes precision / recall / F1 / per-CWE recall.
  6. Saves results to data/real_eval_results.json.

USAGE:
  source venv/bin/activate
  python scripts/real_eval.py                  # 60 vuln + 60 safe
  python scripts/real_eval.py --n-vulnerable 30  # smaller, faster run
  python scripts/real_eval.py --dry-run        # download + filter, no agent
  python scripts/real_eval.py --list-cwes      # show available CWEs in dataset

FREE TIER COST:
  60 vulnerable + 60 safe = 120 samples × 2 Gemini calls = 240 calls
  Free tier: 1500 RPD — you have 6× headroom. Runtime ~25 min.

DATASET CITATION:
  Bhandari et al., "CVEfixes: Automated Collection of Vulnerabilities
  and Their Fixes from Open-Source Software", PROMISE 2021.
  https://doi.org/10.5281/zenodo.4476563
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Optional

# ── venv check ──────────────────────────────────────────────────────────────
import importlib.util as _ilu
if _ilu.find_spec("google.genai") is None and "--dry-run" not in sys.argv and "--list-cwes" not in sys.argv:
    print(
        "\n❌  google.genai not found — virtualenv not active.\n"
        "    Run:  source venv/bin/activate && python scripts/real_eval.py\n",
        file=sys.stderr,
    )
    sys.exit(1)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tools.parse_diff import ParsedPR, ChangedFunction

# ── CWE → your vulnerability_type mapping ───────────────────────────────────
# CVEfixes uses raw CWE IDs. Map them to the 5 categories your agent knows.

CWE_MAP = {
    # SQL Injection
    "CWE-89":  ("SQL_INJECTION",           "HIGH", "REQUEST_CHANGES"),
    "CWE-564": ("SQL_INJECTION",           "HIGH", "REQUEST_CHANGES"),  # SQL via HQL
    # Command Injection
    "CWE-78":  ("CMD_INJECTION",           "HIGH", "REQUEST_CHANGES"),
    "CWE-77":  ("CMD_INJECTION",           "HIGH", "REQUEST_CHANGES"),
    "CWE-88":  ("CMD_INJECTION",           "MEDIUM","REQUEST_CHANGES"),
    # Path Traversal
    "CWE-22":  ("PATH_TRAVERSAL",          "HIGH", "REQUEST_CHANGES"),
    "CWE-23":  ("PATH_TRAVERSAL",          "HIGH", "REQUEST_CHANGES"),
    "CWE-36":  ("PATH_TRAVERSAL",          "MEDIUM","REQUEST_CHANGES"),
    # Insecure Deserialization
    "CWE-502": ("INSECURE_DESERIALIZATION","HIGH", "REQUEST_CHANGES"),
    "CWE-915": ("INSECURE_DESERIALIZATION","HIGH", "REQUEST_CHANGES"),
    # Hardcoded Secrets / Credentials
    "CWE-798": ("HARDCODED_SECRET",        "HIGH", "REQUEST_CHANGES"),
    "CWE-259": ("HARDCODED_SECRET",        "HIGH", "REQUEST_CHANGES"),
    "CWE-321": ("HARDCODED_SECRET",        "HIGH", "REQUEST_CHANGES"),
    # XSS (bonus category — agent can catch these too)
    "CWE-79":  ("XSS",                     "HIGH", "REQUEST_CHANGES"),
    "CWE-80":  ("XSS",                     "MEDIUM","REQUEST_CHANGES"),
}

SUPPORTED_CWES = set(CWE_MAP.keys())

# Min source length — too-short functions give the agent nothing to work with
MIN_SOURCE_CHARS = 80
MAX_SOURCE_CHARS = 4000  # avoid token overload


# ── Dataset loader ───────────────────────────────────────────────────────────

def load_cvefixes_samples(n_vulnerable: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    Load vulnerable + safe (fixed) Python function pairs from CVEfixes.

    Returns:
        (vulnerable_samples, safe_samples) — each is a list of dicts with
        keys matching evaluate.py's benchmark schema.
    """
    print("📥  Loading CVEfixes dataset from HuggingFace (first run downloads ~200 MB)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌  `datasets` not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    # streaming=True: processes records one-at-a-time with ZERO disk caching.
    # Critical when disk space is low — never writes an Arrow file.
    ds = load_dataset(
        "hitoshura25/cvefixes",
        split="train",
        trust_remote_code=True,
        streaming=True,
    )
    print("    Streaming mode: no disk cache needed.")

    vulnerable: list[dict] = []
    safe: list[dict] = []

    seen_ids: set[str] = set()

    for record in ds:
        # Filter to Python only
        lang = (record.get("programming_language") or record.get("language") or "").lower()
        if lang not in ("python", "py"):
            continue

        cwe_id = (record.get("cwe_id") or record.get("cwe") or "").upper().strip()
        # Normalise "CWE89" → "CWE-89" style
        if cwe_id and "-" not in cwe_id:
            cwe_id = cwe_id.replace("CWE", "CWE-")

        if cwe_id not in SUPPORTED_CWES:
            continue

        vuln_type, risk, verdict = CWE_MAP[cwe_id]

        # Extract before/after source — field names vary by dataset version
        before = (
            record.get("vulnerable_code")
            or record.get("before_fix")
            or record.get("vul_func_with_fix")
            or ""
        ).strip()
        after = (
            record.get("fixed_code")
            or record.get("after_fix")
            or record.get("func_after")
            or ""
        ).strip()

        cve_id = record.get("cve_id") or record.get("cve") or "UNKNOWN"
        func_name = record.get("func_name") or record.get("function_name") or "unknown_function"
        repo = record.get("repo") or record.get("project") or "unknown/repo"

        # Dedup by CVE + function name
        uid = f"{cve_id}::{func_name}"
        if uid in seen_ids:
            continue
        seen_ids.add(uid)

        # Quality filter
        if len(before) < MIN_SOURCE_CHARS or len(before) > MAX_SOURCE_CHARS:
            continue
        if len(after) < MIN_SOURCE_CHARS or len(after) > MAX_SOURCE_CHARS:
            continue

        sample_id_base = f"{cve_id}_{func_name}".replace("/", "_").replace(" ", "_")[:60]

        # Vulnerable sample (before-fix code)
        vulnerable.append({
            "id":               f"real_vuln_{sample_id_base}",
            "function_name":    func_name,
            "file_path":        f"{repo}/vuln.py",
            "function_source":  before,
            "expected_verdict": verdict,
            "expected_risk":    risk,
            "vulnerability_type": vuln_type,
            "cwe":              cwe_id,
            "cve":              cve_id,
            "repo":             repo,
            "description":      f"{vuln_type} in {func_name} ({cve_id}) — before patch",
            "source": "cvefixes",
        })

        # Safe counterpart (after-fix code)
        safe.append({
            "id":               f"real_safe_{sample_id_base}",
            "function_name":    func_name,
            "file_path":        f"{repo}/fixed.py",
            "function_source":  after,
            "expected_verdict": "APPROVE",
            "expected_risk":    "LOW",
            "vulnerability_type": "NONE",
            "cwe":              None,
            "cve":              cve_id,
            "repo":             repo,
            "description":      f"Fixed version of {func_name} ({cve_id}) — after patch",
            "source": "cvefixes",
        })

    print(f"    Found {len(vulnerable):,} qualifying vulnerable / {len(safe):,} safe pairs.")

    if not vulnerable:
        print(
            "\n⚠️  No records matched. The dataset field names may differ from expected.\n"
            "   Run with --list-cwes to inspect available fields and CWEs.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Sample balanced set
    rng = random.Random(seed)
    n = min(n_vulnerable, len(vulnerable), len(safe))
    chosen_vuln = rng.sample(vulnerable, n)
    chosen_safe = rng.sample(safe, n)

    # Balance across CWE categories as much as possible
    chosen_vuln = _balance_by_category(chosen_vuln, n)

    print(f"    Sampled {len(chosen_vuln)} vulnerable + {len(chosen_safe)} safe = {len(chosen_vuln)+len(chosen_safe)} total samples.")
    return chosen_vuln, chosen_safe


def _balance_by_category(samples: list[dict], n: int) -> list[dict]:
    """Try to pick an even spread across vulnerability types."""
    by_cat: dict[str, list] = {}
    for s in samples:
        by_cat.setdefault(s["vulnerability_type"], []).append(s)

    n_cats = len(by_cat)
    per_cat = max(1, n // n_cats)

    balanced = []
    for cat_samples in by_cat.values():
        balanced.extend(cat_samples[:per_cat])

    # Top up to n if needed
    remaining = [s for s in samples if s not in balanced]
    random.shuffle(remaining)
    balanced.extend(remaining[:n - len(balanced)])
    return balanced[:n]


def list_cwes_streaming() -> None:
    """Print CWE distribution in the dataset for debugging (streaming, no disk cache)."""
    from datasets import load_dataset
    from collections import Counter
    print("📥  Streaming dataset to inspect CWEs (no disk cache)...")
    ds = load_dataset("hitoshura25/cvefixes", split="train", trust_remote_code=True, streaming=True)
    cwes = Counter()
    langs = Counter()
    for record in ds:
        lang = (record.get("programming_language") or record.get("language") or "unknown").lower()
        langs[lang] += 1
        cwe = (record.get("cwe_id") or record.get("cwe") or "unknown").upper()
        cwes[cwe] += 1
    print("\nTop 30 CWEs in dataset:")
    for cwe, count in cwes.most_common(30):
        tag = " ✓ SUPPORTED" if cwe in SUPPORTED_CWES else ""
        print(f"  {cwe:<12} {count:>6}{tag}")
    print("\nTop languages:")
    for lang, count in langs.most_common(10):
        print(f"  {lang:<20} {count:>6}")


# ── Reuse evaluate.py helpers ────────────────────────────────────────────────

def _build_fake_pr(sample: dict, pr_number: int) -> ParsedPR:
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
        pr_title=f"[REAL-EVAL] {sample['id']}",
        pr_description=sample.get("description", ""),
        repo_full_name=sample.get("repo", "eval/real"),
        base_branch="main",
        head_branch="eval",
        changed_functions=[changed_fn],
        changed_files=[sample["file_path"]],
        total_additions=n_lines,
        total_deletions=0,
    )


def _is_flagged(verdict: str) -> bool:
    return verdict == "REQUEST_CHANGES"


def _compute_metrics(results: list[dict]) -> dict:
    truly_vulnerable = [r for r in results if _is_flagged(r["expected_verdict"])]
    truly_safe       = [r for r in results if not _is_flagged(r["expected_verdict"])]
    tp = [r for r in truly_vulnerable if _is_flagged(r["actual_verdict"])]
    fn = [r for r in truly_vulnerable if not _is_flagged(r["actual_verdict"])]
    fp = [r for r in truly_safe if _is_flagged(r["actual_verdict"])]
    tn = [r for r in truly_safe if not _is_flagged(r["actual_verdict"])]

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 1.0
    recall    = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 1.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    risk_correct  = sum(1 for r in truly_vulnerable if r.get("actual_risk") == r.get("expected_risk"))
    risk_accuracy = risk_correct / len(truly_vulnerable) if truly_vulnerable else 0.0

    categories: dict[str, dict] = {}
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
            "cwe":    next((r["cwe"] for r in truly_vulnerable if r["vulnerability_type"] == cat), None),
        }
        for cat, v in categories.items()
    }

    return {
        "total":         len(results),
        "vulnerable":    len(truly_vulnerable),
        "safe":          len(truly_safe),
        "tp":            len(tp),
        "fp":            len(fp),
        "fn":            len(fn),
        "tn":            len(tn),
        "precision":     round(precision, 3),
        "recall":        round(recall, 3),
        "f1":            round(f1, 3),
        "risk_accuracy": round(risk_accuracy, 3),
        "per_category":  per_cat,
    }


def _print_report(metrics: dict, results: list[dict], elapsed: float, gemini_calls: int,
                   dataset_label: str = "CVEfixes (real CVEs)"):
    W = 62
    print(f"\n{'═'*W}")
    print(f"  AI Code Reviewer — Real-World CVE Evaluation")
    print(f"  Dataset : {dataset_label}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*W}")
    print(f"\n  Samples : {metrics['total']}  "
          f"({metrics['vulnerable']} real CVEs + {metrics['safe']} patched-safe)")
    print(f"  Elapsed : {elapsed:.1f}s   Gemini calls: {gemini_calls}\n")
    sep = "─" * W
    print(sep)
    print(f"  {'Metric':<24} {'Value':>10}   {'Raw':>14}")
    print(sep)
    print(f"  {'Precision':<24} {metrics['precision']:>10.1%}   "
          f"  {metrics['tp']} TP / {metrics['tp']+metrics['fp']} flagged")
    print(f"  {'Recall':<24} {metrics['recall']:>10.1%}   "
          f"  {metrics['tp']} caught / {metrics['vulnerable']} CVEs")
    print(f"  {'F1 Score':<24} {metrics['f1']:>10.1%}")
    print(f"  {'Risk level accuracy':<24} {metrics['risk_accuracy']:>10.1%}   "
          f"  {round(metrics['risk_accuracy']*metrics['vulnerable'])} / {metrics['vulnerable']} correct")
    print(sep)

    if metrics["per_category"]:
        print(f"\n  Per-CWE recall:")
        for cat, v in sorted(metrics["per_category"].items()):
            bar = "█" * int(v["recall"] * 20) + "░" * (20 - int(v["recall"] * 20))
            cwe = f"({v['cwe']})" if v.get("cwe") else ""
            print(f"    {cat:<30} {bar}  {v['recall']:>5.0%}  ({v['caught']}/{v['total']}) {cwe}")

    fn_rows = [r for r in results if _is_flagged(r["expected_verdict"]) and not _is_flagged(r["actual_verdict"])]
    fp_rows = [r for r in results if not _is_flagged(r["expected_verdict"]) and _is_flagged(r["actual_verdict"])]

    if fn_rows:
        print(f"\n  ✗ False negatives — missed real CVEs ({len(fn_rows)}):")
        for r in fn_rows[:10]:
            print(f"    · {r['id'][:55]}  →  {r['actual_verdict']}")
    if fp_rows:
        print(f"\n  ⚠ False positives — patched code re-flagged ({len(fp_rows)}):")
        for r in fp_rows[:10]:
            print(f"    · {r['id'][:55]}  →  {r['actual_verdict']}")
    if not fn_rows and not fp_rows:
        print(f"\n  ✓ Perfect score!")

    print(f"\n{'═'*W}\n")
    print("📋  Resume-ready sentence:")
    print(f"    \"Validated on {metrics['vulnerable']} real CVE-patched Python functions")
    print(f"    (CVEfixes dataset, {', '.join(sorted(metrics['per_category'].keys()))}):")
    print(f"    {metrics['recall']:.0%} recall, {metrics['precision']:.0%} precision, {metrics['f1']:.0%} F1.\"")
    print()


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_real_evaluation(n_vulnerable: int = 60, seed: int = 42,
                        dry_run: bool = False, output_path: Optional[str] = None) -> dict:
    vulnerable_samples, safe_samples = load_cvefixes_samples(n_vulnerable, seed=seed)
    all_samples = vulnerable_samples + safe_samples
    random.Random(seed).shuffle(all_samples)

    if dry_run:
        print(f"\n✓ Dry run complete. {len(all_samples)} samples validated. No agent calls made.")
        # Show category breakdown
        from collections import Counter
        cats = Counter(s["vulnerability_type"] for s in vulnerable_samples)
        print("\nCategory breakdown:")
        for cat, count in sorted(cats.items()):
            cwe = next((s["cwe"] for s in vulnerable_samples if s["vulnerability_type"] == cat), "?")
            print(f"  {cat:<30} {count:>3} samples  ({cwe})")
        return {"dry_run": True, "samples": len(all_samples)}

    from agent.react_agent import run_agent_review, agent_review_to_dict

    results = []
    total_gemini_calls = 0
    start_time = time.time()

    for i, sample in enumerate(all_samples):
        label = "VULN" if sample["expected_verdict"] == "REQUEST_CHANGES" else "SAFE"
        print(f"[{i+1:03d}/{len(all_samples)}] [{label}] {sample['id'][:50]} ...", end=" ", flush=True)

        fake_pr = _build_fake_pr(sample, pr_number=2000 + i)

        try:
            review = run_agent_review(parsed_pr=fake_pr, complexity_results={}, repo_path=None)
            review_dict = agent_review_to_dict(review)

            actual_verdict = review_dict.get("overall_verdict", "COMMENT")
            actual_risk    = review_dict.get("risk_level", "LOW")
            gemini_calls   = review_dict.get("agent_meta", {}).get("gemini_calls", 2)
            total_gemini_calls += gemini_calls

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

            results.append({
                "id":                   sample["id"],
                "expected_verdict":     sample["expected_verdict"],
                "expected_risk":        sample["expected_risk"],
                "actual_verdict":       actual_verdict,
                "actual_risk":          actual_risk,
                "vulnerability_type":   sample["vulnerability_type"],
                "cwe":                  sample.get("cwe"),
                "cve":                  sample.get("cve"),
                "repo":                 sample.get("repo"),
                "description":          sample.get("description", ""),
                "summary":              review_dict.get("summary", ""),
                "confidence":           review_dict.get("confidence", 0.0),
                "tools_used":           review_dict.get("tools_used", []),
                "gemini_calls":         gemini_calls,
                "outcome":              outcome,
            })
            print(f"{outcome}  ({actual_verdict} / {actual_risk})")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "id":                   sample["id"],
                "expected_verdict":     sample["expected_verdict"],
                "expected_risk":        sample["expected_risk"],
                "actual_verdict":       "ERROR",
                "actual_risk":          "UNKNOWN",
                "vulnerability_type":   sample["vulnerability_type"],
                "outcome":              "ERR",
                "error":                str(e),
            })

    elapsed = time.time() - start_time
    clean_results = [r for r in results if r.get("outcome") != "ERR"]
    metrics = _compute_metrics(clean_results)

    _print_report(metrics, clean_results, elapsed, total_gemini_calls)

    if output_path is None:
        output_path = os.path.join(_ROOT, "data", "real_eval_results.json")

    output = {
        "run_timestamp":   datetime.utcnow().isoformat() + "Z",
        "dataset":         "CVEfixes (HuggingFace: hitoshura25/cvefixes)",
        "n_vulnerable":    len(vulnerable_samples),
        "n_safe":          len(safe_samples),
        "elapsed_seconds": round(elapsed, 1),
        "gemini_calls":    total_gemini_calls,
        "metrics":         metrics,
        "results":         results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved → {output_path}")
    return output


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AI Code Reviewer on real CVE-patched functions (CVEfixes dataset)."
    )
    parser.add_argument(
        "--n-vulnerable", type=int, default=60,
        help="Number of real vulnerable functions to test (default: 60). Equal # of safe added."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Download + filter dataset only — no agent/Gemini calls"
    )
    parser.add_argument(
        "--list-cwes", action="store_true",
        help="Show CWE distribution in the CVEfixes dataset and exit"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: data/real_eval_results.json)"
    )
    args = parser.parse_args()

    if args.list_cwes:
        list_cwes_streaming()
        return

    run_real_evaluation(
        n_vulnerable=args.n_vulnerable,
        seed=args.seed,
        dry_run=args.dry_run,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
