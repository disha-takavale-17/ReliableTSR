"""Unit tests for the repair evaluation script."""
import json
from pathlib import Path
from scripts.eval_repair import run_evaluation


def test_eval_repair_mock_smoke(tmp_path):
    out_file = tmp_path / "mock_eval.json"
    report = run_evaluation(
        results_dir="data/results",
        cache_dir=str(tmp_path / "cache"),
        limit=2,
        mock=True,
        group_filter="development",
    )
    assert "development" in report["groups"]
    dev = report["groups"]["development"]
    assert dev["evaluated_tables"] == 2
    assert dev["cell_metrics"]["total_flagged_cells"] > 0
    assert dev["tables_with_suspicious_span_or_empty_grid"] == 34
    assert dev["eligible_tables_with_flagged_cells"] == 58
