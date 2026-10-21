"""Development-only threshold selection and untouched final-group evaluation."""
import json
from collections import Counter
from pathlib import Path

from .risk import DEFAULT_WEIGHTS, assess, confidence_baseline


def metrics(labels, scores, threshold):
    tp = sum(bool(y) and s >= threshold for y, s in zip(labels, scores))
    fp = sum(not y and s >= threshold for y, s in zip(labels, scores))
    fn = sum(bool(y) and s < threshold for y, s in zip(labels, scores))
    tn = sum(not y and s < threshold for y, s in zip(labels, scores))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(precision, 4),
            "recall": round(recall, 4), "false_alarm_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0}


def tune(labels, scores):
    candidates = sorted({0.0, 1.00001, *scores})
    return max(candidates, key=lambda t: (metrics(labels, scores, t)["f1"],
                                           metrics(labels, scores, t)["precision"], -t))


def evaluate(results_dir):
    files = sorted(Path(results_dir).rglob("*.json"))
    records = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    records = [r for r in records if r.get("schema_version") == "1.0" and "comparison" in r]
    dev = [r for r in records if r["group"] == "development"]
    final = [r for r in records if r["group"] == "final"]
    if not dev or not final:
        raise ValueError("Need processed development and final groups")
    def score(r, enabled=None, floor=0.7):
        return assess(r["prediction"], r["raw_detections"], r["image_size"],
                      enabled=enabled, confidence_floor=floor)["score"]
    y_dev = [r["comparison"]["has_error"] for r in dev]
    y_final = [r["comparison"]["has_error"] for r in final]
    candidates = []
    for floor in (0.7, 0.85, 0.95, 0.99):
        dev_scores = [score(r, floor=floor) for r in dev]
        threshold = tune(y_dev, dev_scores)
        m = metrics(y_dev, dev_scores, threshold)
        candidates.append((m["f1"], m["precision"], -floor, floor, threshold))
    _, _, _, confidence_floor, risk_threshold = max(candidates)
    baseline_threshold = tune(y_dev, [confidence_baseline(r["raw_detections"]) for r in dev])
    risk_scores = [score(r, floor=confidence_floor) for r in final]
    baseline_scores = [confidence_baseline(r["raw_detections"]) for r in final]
    by_type = {}
    for kind in sorted({t for r in final for t in r["comparison"]["error_types"]}):
        subset = [i for i, r in enumerate(final) if kind in r["comparison"]["error_types"]]
        by_type[kind] = {"tables": len(subset),
                         "risk_recall": round(sum(risk_scores[i] >= risk_threshold for i in subset) / len(subset), 4),
                         "confidence_recall": round(sum(baseline_scores[i] >= baseline_threshold for i in subset) / len(subset), 4)}
    ablations = {}
    for signal in DEFAULT_WEIGHTS:
        enabled = set(DEFAULT_WEIGHTS) - {signal}
        removed = [score(r, enabled, confidence_floor) for r in final]
        ablations[signal] = {"mean_score_contribution": round(sum(score(r, floor=confidence_floor) - removed[i] for i, r in enumerate(final)) / len(final), 4),
                             "tables_triggered": sum(any(x["signal"] == signal for x in assess(r["prediction"], r["raw_detections"], r["image_size"], confidence_floor=confidence_floor)["reasons"]) for r in final),
                             "f1_without_signal_at_fixed_threshold": metrics(y_final, removed, risk_threshold)["f1"]}
    report = {"schema_version": "1.0", "status": "measured", "development_tables": len(dev),
              "final_tables": len(final), "development_tuning": {"risk_threshold": risk_threshold,
              "confidence_floor": confidence_floor,
              "confidence_threshold": baseline_threshold, "objective": "maximum table-level F1; tie-break precision"},
              "final_risk": metrics(y_final, risk_scores, risk_threshold),
              "final_confidence_baseline": metrics(y_final, baseline_scores, baseline_threshold),
              "by_error_type": by_type, "signal_contribution": ablations,
              "risk_weights": DEFAULT_WEIGHTS,
              "label_definition": "any error evidence from one-to-one cell matching",
              "caveat": "Pilot subset of official test split; thresholds selected on development subset only."}
    return report
