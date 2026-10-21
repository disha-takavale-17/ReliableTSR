"""Prediction-only warnings. This module accepts no GT data."""
from statistics import mean

from .geometry import area, intersection

DEFAULT_WEIGHTS = {"low_confidence": 0.24, "cell_overlap": 0.21,
                   "grid_gap": 0.17, "axis_conflict": 0.15,
                   "suspicious_span": 0.15, "empty_grid": 0.08}


def signals(pred, detections, image_size, confidence_floor=0.7, detection_threshold=0.5):
    cells = pred["cells"]
    rows, cols = pred["rows"], pred["columns"]
    findings = []

    def add(kind, severity, ids, reason):
        findings.append({"signal": kind, "severity": round(min(1.0, severity), 4),
                         "cells": sorted(set(ids)), "reason": reason})

    structural = [d for d in detections if d["score"] >= detection_threshold and
                  d["label"] in ("table row", "table column", "table spanning cell",
                                 "table column header", "table projected row header")]
    weak = [d for d in structural if d["score"] < confidence_floor]
    if structural:
        average = mean(d["score"] for d in structural)
        severity = min(1.0, 0.5 * (1 - average) / 0.15 +
                       0.5 * len(weak) / len(structural))
        add("low_confidence", severity,
            [c["id"] for c in cells if c["score"] < confidence_floor],
            f"Mean structural confidence is {average:.2f}; {len(weak)} detections are below {confidence_floor:.2f}.")
    if not rows or not cols:
        add("empty_grid", 1, [], "No complete row and column grid was detected.")
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            overlap = area(intersection(a["bbox"], b["bbox"]))
            fraction = overlap / min(area(a["bbox"]), area(b["bbox"])) if min(area(a["bbox"]), area(b["bbox"])) else 0
            if fraction > 0.15:
                add("cell_overlap", fraction, [a["id"], b["id"]],
                    f"Cells {a['id']} and {b['id']} overlap by {fraction:.0%} of the smaller cell.")
    for axis, items, start, end in (("row", rows, 1, 3), ("column", cols, 0, 2)):
        if len(items) < 2:
            continue
        sizes = [x["bbox"][end] - x["bbox"][start] for x in items]
        typical = max(1.0, mean(sizes))
        for i, (a, b) in enumerate(zip(items, items[1:])):
            gap = b["bbox"][start] - a["bbox"][end]
            if gap > 0.25 * typical:
                add("grid_gap", min(1, gap / typical),
                    [c["id"] for c in cells if i in (c["rows"] if axis == "row" else c["cols"])],
                    f"A {gap:.1f}px gap separates adjacent {axis}s.")
            if gap < -0.3 * typical:
                add("axis_conflict", min(1, -gap / typical),
                    [c["id"] for c in cells if i in (c["rows"] if axis == "row" else c["cols"])],
                    f"Adjacent {axis}s overlap substantially.")
    for d in pred["diagnostics"]:
        if d["type"] in ("irregular_span", "overlapping_spans", "unmapped_span", "single_slot_span"):
            add("suspicious_span", 0.8, [], f"Span detection {d['source']} is {d['type'].replace('_', ' ')}.")
    return findings


def assess(pred, detections, image_size, weights=None, threshold=0.4, enabled=None,
           confidence_floor=0.7):
    """Return a table-level score and localized possible-error explanations."""
    weights = weights or DEFAULT_WEIGHTS
    enabled = set(enabled or weights)
    findings = [f for f in signals(pred, detections, image_size,
                                   confidence_floor=confidence_floor) if f["signal"] in enabled]
    contributions = {name: round(weights[name] * max((f["severity"] for f in findings
                                                       if f["signal"] == name), default=0), 4)
                     for name in weights}
    score = round(min(1.0, sum(contributions.values())), 4)
    return {"score": score, "threshold": threshold, "confidence_floor": confidence_floor,
            "weights": weights, "possible_error": score >= threshold,
            "affected_cells": sorted({c for f in findings for c in f["cells"]}),
            "reasons": findings, "contributions": contributions}


def confidence_baseline(detections, detection_threshold=0.5):
    scores = [d["score"] for d in detections if d["score"] >= detection_threshold and d["label"] in ("table row", "table column",
              "table spanning cell", "table column header", "table projected row header")]
    return round(1 - mean(scores), 4) if scores else 1.0
