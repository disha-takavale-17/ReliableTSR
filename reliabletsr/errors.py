"""One-to-one IoU matching and evidence-based structural error labels."""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import coverage, iou


def _match_indices(matrix, min_iou, ambiguity_delta):
    """Maximize accepted matches first, then maximize their total IoU."""
    if not matrix or not matrix[0]:
        return []

    overlaps = np.asarray(matrix, dtype=float)
    bonus = max(overlaps.shape) + 1
    reward = np.where(overlaps >= min_iou, bonus + overlaps, 0.0)
    gt_indices, pred_indices = linear_sum_assignment(-reward)
    matches = []
    for i, j in zip(gt_indices.tolist(), pred_indices.tolist()):
        score = float(overlaps[i, j])
        if score < min_iou:
            continue
        rivals = [float(overlaps[i, k]) for k in range(overlaps.shape[1])
                  if k != j and overlaps[i, k] >= min_iou]
        rivals += [float(overlaps[k, j]) for k in range(overlaps.shape[0])
                   if k != i and overlaps[k, j] >= min_iou]
        matches.append((i, j, score,
                        bool(rivals and score - max(rivals) <= ambiguity_delta)))
    return matches


def _cell_keys(error):
    """A GT cell ID and the same prediction cell ID are different cells."""
    return {("gt", ident) for ident in error["gt_cells"]} | {
        ("pred", ident) for ident in error["pred_cells"]
    }


def compare(gt, pred, min_iou=0.5, ambiguity_delta=0.05):
    if not 0 < min_iou <= 1:
        raise ValueError("min_iou must be in (0, 1]")
    if ambiguity_delta < 0:
        raise ValueError("ambiguity_delta must be nonnegative")

    gt_cells, pred_cells = gt["cells"], pred["cells"]
    matrix = [[iou(a["bbox"], b["bbox"]) for b in pred_cells]
              for a in gt_cells]
    matches = [
        {"gt": gt_cells[i]["id"], "pred": pred_cells[j]["id"],
         "iou": round(score, 4), "ambiguous": ambiguous}
        for i, j, score, ambiguous in _match_indices(matrix, min_iou,
                                                       ambiguity_delta)
    ]
    by_gt = {m["gt"] for m in matches}
    by_pred = {m["pred"] for m in matches}
    gt_by_id = {cell["id"]: cell for cell in gt_cells}
    pred_by_id = {cell["id"]: cell for cell in pred_cells}
    errors = []

    def record(kind, gt_ids, pred_ids, evidence, ambiguous=False):
        errors.append({"type": kind, "gt_cells": gt_ids, "pred_cells": pred_ids,
                       "evidence": evidence, "ambiguous": bool(ambiguous)})

    for cell in gt_cells:
        if cell["id"] in by_gt:
            continue
        near = [(candidate["id"], round(fraction, 4))
                for candidate in pred_cells
                if (fraction := coverage(cell["bbox"], candidate["bbox"])) >= 0.3]
        record("missing_cell", [cell["id"]], [ident for ident, _ in near],
               {"overlapping_predictions": near}, bool(near))

    for cell in pred_cells:
        if cell["id"] in by_pred:
            continue
        near = [(candidate["id"], round(fraction, 4))
                for candidate in gt_cells
                if (fraction := coverage(cell["bbox"], candidate["bbox"])) >= 0.3]
        record("extra_cell", [ident for ident, _ in near], [cell["id"]],
               {"overlapping_ground_truth": near}, bool(near))

    for match in matches:
        gt_cell = gt_by_id[match["gt"]]
        pred_cell = pred_by_id[match["pred"]]
        gt_span = [len(gt_cell["rows"]), len(gt_cell["cols"])]
        pred_span = [len(pred_cell["rows"]), len(pred_cell["cols"])]
        if gt_span != pred_span:
            record("span_mistake", [gt_cell["id"]], [pred_cell["id"]],
                   {"gt_span": gt_span, "pred_span": pred_span,
                    "iou": match["iou"]}, match["ambiguous"])
        if (gt_cell["header"] != pred_cell["header"] or
                gt_cell["projected_row_header"] != pred_cell["projected_row_header"]):
            record("header_mistake", [gt_cell["id"]], [pred_cell["id"]],
                   {"gt_header": gt_cell["header"],
                    "pred_header": pred_cell["header"],
                    "gt_projected": gt_cell["projected_row_header"],
                    "pred_projected": pred_cell["projected_row_header"]},
                   match["ambiguous"])

    for cell in pred_cells:
        parts = [candidate for candidate in gt_cells
                 if coverage(candidate["bbox"], cell["bbox"]) >= 0.5]
        if len(parts) >= 2 and len(cell["rows"]) * len(cell["cols"]) > 1:
            record("likely_merge", [part["id"] for part in parts], [cell["id"]],
                   {"covered_gt_cells": len(parts)}, True)

    for cell in gt_cells:
        parts = [candidate for candidate in pred_cells
                 if coverage(candidate["bbox"], cell["bbox"]) >= 0.5]
        if len(parts) >= 2 and len(cell["rows"]) * len(cell["cols"]) > 1:
            record("likely_split", [cell["id"]], [part["id"] for part in parts],
                   {"covered_pred_cells": len(parts)}, True)

    labels_by_cell = defaultdict(set)
    for error in errors:
        for key in _cell_keys(error):
            labels_by_cell[key].add(error["type"])
    for error in errors:
        related = set().union(*(labels_by_cell[key] for key in _cell_keys(error)))
        error["cooccurring_labels"] = sorted(related - {error["type"]})

    return {
        "method": "Hungarian assignment maximizing accepted-match count, then total box IoU",
        "min_iou": min_iou,
        "ambiguity_delta": ambiguity_delta,
        "matches": matches,
        "errors": errors,
        "has_error": bool(errors),
        "error_types": sorted({error["type"] for error in errors}),
    }
