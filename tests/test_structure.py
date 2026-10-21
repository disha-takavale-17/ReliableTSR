from pathlib import Path

from reliabletsr.errors import _match_indices, compare
from reliabletsr.risk import assess
from reliabletsr.structure import build_grid, parse_xml


def obj(label, box, ident, score=1):
    return {"id": ident, "label": label, "bbox": box, "score": score}


def base():
    return [obj("table row", [0, 0, 20, 10], "r0"),
            obj("table row", [0, 10, 20, 20], "r1"),
            obj("table column", [0, 0, 10, 20], "k0"),
            obj("table column", [10, 0, 20, 20], "k1")]


def test_blank_slots_and_spanning_cells():
    grid = build_grid(base())
    assert len(grid["cells"]) == 4  # blank slots are still cells
    span = build_grid(base() + [obj("table spanning cell", [0, 0, 20, 10], "s")])
    assert len(span["cells"]) == 3
    assert any(c["cols"] == [0, 1] for c in span["cells"])


def test_xml_and_header(tmp_path):
    xml = """<annotation><filename>x.jpg</filename><size><width>20</width><height>20</height></size>
    <object><name>table row</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>20</xmax><ymax>10</ymax></bndbox></object>
    <object><name>table column</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object>
    <object><name>table column header</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>20</xmax><ymax>10</ymax></bndbox></object></annotation>"""
    path = tmp_path / "x.xml"
    path.write_text(xml)
    parsed = parse_xml(path)
    grid = build_grid(parsed["objects"])
    assert grid["cells"][0]["header"] is True


def test_merge_and_multiple_labels():
    gt = build_grid(base())
    pred = build_grid(base() + [obj("table spanning cell", [0, 0, 20, 10], "s")])
    result = compare(gt, pred)
    types = set(result["error_types"])
    assert "likely_merge" in types and "missing_cell" in types
    assert "span_mistake" in types
    merge = next(e for e in result["errors"] if e["type"] == "likely_merge")
    assert {"missing_cell", "span_mistake"} <= set(merge["cooccurring_labels"])


def test_split_and_header_mistake():
    gt = build_grid(base() + [obj("table spanning cell", [0, 0, 20, 10], "s"),
                              obj("table column header", [0, 0, 20, 10], "h")])
    pred = build_grid(base())
    kinds = set(compare(gt, pred)["error_types"])
    assert {"likely_split", "extra_cell", "header_mistake"} <= kinds


def test_risk_never_needs_gt():
    detections = base()
    grid = build_grid(detections)
    result = assess(grid, detections, [20, 20])
    assert 0 <= result["score"] <= 1
    assert "ground_truth" not in result


def test_gt_and_prediction_ids_do_not_create_false_cooccurrence():
    gt = {"cells": [{"id": "c0", "bbox": [0, 0, 10, 10], "rows": [0],
                     "cols": [0], "header": False, "projected_row_header": False}]}
    pred = {"cells": [{"id": "c0", "bbox": [20, 20, 30, 30], "rows": [0],
                       "cols": [0], "header": False, "projected_row_header": False}]}
    errors = compare(gt, pred)["errors"]
    assert {e["type"] for e in errors} == {"missing_cell", "extra_cell"}
    assert all(not e["cooccurring_labels"] for e in errors)


def test_assignment_prefers_two_accepted_pairs_over_higher_raw_iou():
    overlaps = [[0.90, 0.51], [0.50, 0.49]]
    matches = _match_indices(overlaps, min_iou=0.5, ambiguity_delta=0.05)
    assert {(i, j) for i, j, _, _ in matches} == {(0, 1), (1, 0)}


def test_below_threshold_rival_is_not_ambiguous():
    matches = _match_indices([[0.51, 0.49]], min_iou=0.5,
                             ambiguity_delta=0.05)
    assert matches == [(0, 0, 0.51, False)]
