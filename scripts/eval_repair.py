"""Evaluate Qwen visual repair step on labelled benchmark tables.

For each flagged cell in development and final splits:
- Excludes tables flagged only by suspicious_span and empty_grid.
- Submits table crop with numbered bounding boxes to Qwen (or reads from disk cache).
- Parses and validates JSON edit proposals.
- Applies verified repairs to predicted cell grid (rejecting non-rectangular or overlapping edits).
- Reruns errors.py one-to-one matching vs ground truth XML annotations.
- Scores cell-level outcomes:
    - fixed: TATR error successfully corrected
    - unchanged: TATR error persists or false alarm retained without cancellation
    - broken: Originally correct TATR cell damaged by repair
    - false_alarm_cancelled: Flagged false alarm verified as benign (no_change)
- Measures invalid-JSON count, Qwen API calls, and excluded tables.
- Reports development and final groups separately.
"""
import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from reliabletsr.errors import compare
from reliabletsr.qwen.client import FREE_OPENROUTER_MODEL, QwenClient
from reliabletsr.qwen.repair import (
    QwenRepairAuditor,
    get_eligible_flagged_cells,
    validate_repair_response,
)
from reliabletsr.repair_apply import apply_repairs


def evaluate_table_repair(
    record: Dict[str, Any],
    auditor: QwenRepairAuditor,
    mock: bool = False,
) -> Dict[str, Any]:
    """Evaluate repair step on a single table record."""
    table_id = Path(record["image"]).stem
    image_path = Path(record["image"])
    gt = record["ground_truth"]["grid"]
    original_pred = record["prediction"]
    risk = record["risk"]
    min_iou = record["settings"].get("min_iou", 0.5)

    flagged_cells, cell_signals, excluded_only = get_eligible_flagged_cells(original_pred, risk)
    if not flagged_cells:
        return {
            "table_id": table_id,
            "eligible": False,
            "excluded_only": excluded_only,
            "flagged_cell_count": 0,
            "fixed": 0,
            "unchanged": 0,
            "broken": 0,
            "false_alarm_cancelled": 0,
            "invalid_json_count": 0,
            "qwen_called": False,
            "missing": False,
        }

    flagged_ids = [c["id"] for c in flagged_cells]

    # Baseline comparison vs GT before repair
    orig_comparison = compare(gt, original_pred, min_iou=min_iou)
    orig_error_cells: Set[str] = {
        cid for err in orig_comparison.get("errors", []) for cid in err.get("pred_cells", [])
    }

    # Obtain repair proposals via Qwen or mock
    if mock:
        proposals = []
        for c in flagged_cells:
            cid = c["id"]
            if cid in orig_error_cells:
                # Mock: try to fix error
                proposals.append({
                    "cell_id": cid,
                    "verdict": "REAL_ERROR",
                    "action": "no_change",
                    "params": {},
                })
            else:
                # Mock: cancel false alarm
                proposals.append({
                    "cell_id": cid,
                    "verdict": "FALSE_ALARM",
                    "action": "no_change",
                    "params": {},
                })
        invalid_count = 0
        qwen_called = False
    else:
        if not image_path.exists():
            return {
                "table_id": table_id,
                "eligible": True,
                "excluded_only": False,
                "missing": True,
                "missing_reason": f"Image file not found: {image_path}",
                "flagged_cell_count": len(flagged_ids),
            }

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        try:
            audit_result = auditor.audit_and_repair(image, original_pred, risk, table_id=table_id)
        except Exception as exc:
            return {
                "table_id": table_id,
                "eligible": True,
                "excluded_only": False,
                "missing": True,
                "missing_reason": str(exc),
                "flagged_cell_count": len(flagged_ids),
                "invalid_json_count": 0,
                "qwen_called": False,
            }

        proposals = audit_result["proposals"]
        invalid_count = audit_result["invalid_count"]
        qwen_called = audit_result["qwen_called"]

    # Apply repairs to prediction grid
    repaired_pred, exec_log = apply_repairs(original_pred, proposals)

    # Re-evaluate repaired grid against GT
    repaired_comparison = compare(gt, repaired_pred, min_iou=min_iou)
    repaired_error_cells: Set[str] = {
        cid for err in repaired_comparison.get("errors", []) for cid in err.get("pred_cells", [])
    }

    # Evaluate each flagged cell
    fixed_count = 0
    unchanged_count = 0
    broken_count = 0
    false_alarm_cancelled_count = 0

    proposals_by_id = {p["cell_id"]: p for p in proposals}

    for cell in flagged_cells:
        cid = cell["id"]
        prop = proposals_by_id.get(cid, {})
        verdict = prop.get("verdict", "FALSE_ALARM")
        was_in_error = cid in orig_error_cells
        now_in_error = cid in repaired_error_cells

        if was_in_error:
            # TATR originally made an error here
            if not now_in_error and len(repaired_comparison.get("errors", [])) < len(orig_comparison.get("errors", [])):
                fixed_count += 1
            else:
                unchanged_count += 1
        else:
            # TATR originally had no error on this cell (False Alarm)
            if now_in_error or len(repaired_comparison.get("errors", [])) > len(orig_comparison.get("errors", [])):
                broken_count += 1
            elif verdict == "FALSE_ALARM":
                false_alarm_cancelled_count += 1
            else:
                unchanged_count += 1

    return {
        "table_id": table_id,
        "eligible": True,
        "excluded_only": False,
        "missing": False,
        "flagged_cell_count": len(flagged_cells),
        "orig_errors": len(orig_comparison.get("errors", [])),
        "repaired_errors": len(repaired_comparison.get("errors", [])),
        "fixed": fixed_count,
        "unchanged": unchanged_count,
        "broken": broken_count,
        "false_alarm_cancelled": false_alarm_cancelled_count,
        "invalid_json_count": invalid_count,
        "qwen_called": qwen_called,
        "proposals": proposals,
        "execution_log": exec_log,
    }


def run_evaluation(
    results_dir: str = "data/results",
    cache_dir: str = "data/repair_cache",
    limit: Optional[int] = None,
    api_key: Optional[str] = None,
    model: str = FREE_OPENROUTER_MODEL,
    provider: str = "openrouter",
    mock: bool = False,
    group_filter: str = "both",
) -> Dict[str, Any]:
    """Run full repair evaluation across development and final splits."""
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    client = None
    if not mock and (key or provider == "ollama"):
        client = QwenClient(api_key=key, provider=provider, model=model)

    auditor = QwenRepairAuditor(client=client, cache_dir=cache_dir)

    groups = ["development", "final"] if group_filter == "both" else [group_filter]
    overall_report: Dict[str, Any] = {
        "schema_version": "2.0-repair-eval",
        "model": model if not mock else "mock-offline",
        "provider": provider if not mock else "mock",
        "groups": {},
    }

    for group in groups:
        group_path = Path(results_dir) / group
        files = sorted(group_path.glob("*.json"))
        total_tables = len(files)

        eligible_records: List[Tuple[Path, Dict[str, Any]]] = []
        span_or_empty_flag_tables = 0
        no_cell_flag_tables = 0

        for f in files:
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            pred = rec.get("prediction", {})
            risk = rec.get("risk", {})
            flagged, _, has_excluded_flags = get_eligible_flagged_cells(pred, risk)
            if has_excluded_flags:
                span_or_empty_flag_tables += 1
            if not flagged:
                no_cell_flag_tables += 1
            else:
                eligible_records.append((f, rec))

        # Apply limit to eligible tables if specified
        eval_records = eligible_records
        if limit is not None and limit > 0:
            eval_records = eligible_records[:limit]

        group_results: List[Dict[str, Any]] = []
        tot_fixed = 0
        tot_unchanged = 0
        tot_broken = 0
        tot_false_alarm_cancelled = 0
        tot_invalid_json = 0
        tot_qwen_calls = 0
        missing_count = 0

        for idx, (path, rec) in enumerate(eval_records, 1):
            res = evaluate_table_repair(rec, auditor, mock=mock)
            group_results.append(res)

            if res.get("missing"):
                missing_count += 1
                continue

            tot_fixed += res.get("fixed", 0)
            tot_unchanged += res.get("unchanged", 0)
            tot_broken += res.get("broken", 0)
            tot_false_alarm_cancelled += res.get("false_alarm_cancelled", 0)
            tot_invalid_json += res.get("invalid_json_count", 0)
            if res.get("qwen_called"):
                tot_qwen_calls += 1

        tot_flagged = tot_fixed + tot_unchanged + tot_broken + tot_false_alarm_cancelled

        overall_report["groups"][group] = {
            "total_tables": total_tables,
            "eligible_tables_with_flagged_cells": len(eligible_records),
            "evaluated_tables": len(eval_records),
            "tables_with_suspicious_span_or_empty_grid": span_or_empty_flag_tables,
            "tables_without_cell_level_flags": no_cell_flag_tables,
            "missing_evaluations": missing_count,
            "cell_metrics": {
                "total_flagged_cells": tot_flagged,
                "fixed": tot_fixed,
                "unchanged": tot_unchanged,
                "broken": tot_broken,
                "false_alarm_cancelled": tot_false_alarm_cancelled,
                "fix_rate": round(tot_fixed / tot_flagged, 4) if tot_flagged else None,
                "false_alarm_cancellation_rate": round(tot_false_alarm_cancelled / tot_flagged, 4) if tot_flagged else None,
                "broken_rate": round(tot_broken / tot_flagged, 4) if tot_flagged else None,
            },
            "system_metrics": {
                "invalid_json_count": tot_invalid_json,
                "qwen_api_calls": tot_qwen_calls,
            },
            "records": group_results,
        }

    return overall_report


def main():
    parser = argparse.ArgumentParser(description="ReliableTSR Qwen Table Repair Evaluator")
    parser.add_argument("--results", default="data/results", help="Directory of saved inference records")
    parser.add_argument("--cache-dir", default="data/repair_cache", help="Disk cache directory for Qwen API outputs")
    parser.add_argument("--out", default="data/repair_evaluation.json", help="Path to write final report")
    parser.add_argument("--limit", type=int, default=None, help="Maximum eligible tables to evaluate per group")
    parser.add_argument("--api-key", default=None, help="OpenRouter API Key (defaults to OPENROUTER_API_KEY env)")
    parser.add_argument("--model", default=FREE_OPENROUTER_MODEL, help="Vision language model identifier")
    parser.add_argument("--provider", default="openrouter", choices=["openrouter", "ollama"], help="AI provider")
    parser.add_argument("--group", default="both", choices=["both", "development", "final"], help="Split to evaluate")
    parser.add_argument("--mock", action="store_true", help="Run offline mock repair without live API calls")

    args = parser.parse_args()

    report = run_evaluation(
        results_dir=args.results,
        cache_dir=args.cache_dir,
        limit=args.limit,
        api_key=args.api_key,
        model=args.model,
        provider=args.provider,
        mock=args.mock,
        group_filter=args.group,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print summary to stdout
    print(f"\n=======================================================")
    print(f"       RELIABLETSR QWEN REPAIR EVALUATION REPORT       ")
    print(f"=======================================================")
    for grp, grp_data in report["groups"].items():
        cm = grp_data["cell_metrics"]
        sm = grp_data["system_metrics"]
        print(f"\n--- Group: {grp.upper()} ---")
        print(f"Total Tables: {grp_data['total_tables']}")
        print(f"Excluded Tables with suspicious_span / empty_grid: {grp_data['tables_with_suspicious_span_or_empty_grid']}")
        print(f"Eligible Tables with Flagged Cells: {grp_data['eligible_tables_with_flagged_cells']}")
        print(f"Evaluated Tables: {grp_data['evaluated_tables']}")
        if grp_data["missing_evaluations"] > 0:
            print(f"Missing / Unmeasurable: {grp_data['missing_evaluations']}")
        print(f"Flagged Cells Evaluated: {cm['total_flagged_cells']}")
        print(f"  * Fixed (Real Errors Corrected): {cm['fixed']}")
        print(f"  * False Alarms Cancelled:       {cm['false_alarm_cancelled']}")
        print(f"  * Unchanged:                    {cm['unchanged']}")
        print(f"  * Broken (Regressions):         {cm['broken']}")
        print(f"System Integrity:")
        print(f"  * Invalid JSON Count:           {sm['invalid_json_count']}")
        print(f"  * Qwen API Calls:               {sm['qwen_api_calls']}")
    print(f"\nFull detailed audit saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
