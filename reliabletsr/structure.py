"""Convert PASCAL VOC objects and TATR detections into an occupied cell grid."""
from pathlib import Path
import xml.etree.ElementTree as ET

from .geometry import area, coverage, intersection, union_box

CLASSES = {"table", "table row", "table column", "table column header",
           "table spanning cell", "table projected row header"}


def parse_xml(path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    detections = []
    for idx, obj in enumerate(root.findall("object")):
        name = obj.findtext("name", "").strip()
        if name not in CLASSES:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        bbox = [float(box.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax")]
        if area(bbox) > 0:
            detections.append({"id": f"gt{idx}", "label": name, "bbox": bbox, "score": 1.0})
    return {"filename": root.findtext("filename", Path(path).stem + ".jpg"),
            "size": [int(size.findtext("width")), int(size.findtext("height"))] if size is not None else None,
            "objects": detections}


def _sorted_axis(objects, name, axis):
    selected = [o for o in objects if o["label"] == name]
    return sorted(selected, key=lambda o: ((o["bbox"][axis] + o["bbox"][axis + 2]) / 2,
                                           o["bbox"][axis]))


def build_grid(objects, threshold=0.5, span_coverage=0.5):
    """Rows/columns form slots; spanning detections replace covered slots.

    Every unclaimed row-column intersection becomes a cell, including blank cells.
    A detected span claims slots when it covers >span_coverage of each slot.
    Overlapping spans are resolved by higher confidence and left as diagnostics.
    """
    active = [o for o in objects if o.get("score", 1) >= threshold and area(o["bbox"]) > 0]
    rows = _sorted_axis(active, "table row", 1)
    cols = _sorted_axis(active, "table column", 0)
    headers = [o for o in active if o["label"] == "table column header"]
    spans = sorted([o for o in active if o["label"] in
                    ("table spanning cell", "table projected row header")],
                   key=lambda o: o.get("score", 1), reverse=True)
    slots = {(r, c): intersection(row["bbox"], col["bbox"])
             for r, row in enumerate(rows) for c, col in enumerate(cols)
             if area(intersection(row["bbox"], col["bbox"])) > 0}
    occupied = set()
    cells = []
    diagnostics = []

    def add_cell(slot_ids, source, score, projected=False):
        rr = sorted({r for r, _ in slot_ids})
        cc = sorted({c for _, c in slot_ids})
        box = union_box([slots[k] for k in slot_ids])
        header = any(coverage(slots[k], h["bbox"]) > 0.5 for k in slot_ids for h in headers)
        cells.append({"id": f"c{len(cells)}", "bbox": box, "rows": rr, "cols": cc,
                      "row_start": min(rr), "row_end": max(rr), "col_start": min(cc),
                      "col_end": max(cc), "header": header, "projected_row_header": projected,
                      "score": float(score), "source": source})

    for span in spans:
        claimed = {k for k, box in slots.items() if coverage(box, span["bbox"]) > span_coverage}
        if not claimed:
            diagnostics.append({"type": "unmapped_span", "source": span["id"]})
            continue
        rr = {r for r, _ in claimed}
        cc = {c for _, c in claimed}
        rectangle = {(r, c) for r in rr for c in cc}
        if claimed != rectangle or rectangle - set(slots):
            diagnostics.append({"type": "irregular_span", "source": span["id"]})
            continue
        if occupied & claimed:
            diagnostics.append({"type": "overlapping_spans", "source": span["id"]})
            continue
        if len(claimed) == 1:
            diagnostics.append({"type": "single_slot_span", "source": span["id"]})
        add_cell(claimed, span["id"], span.get("score", 1),
                 span["label"] == "table projected row header")
        occupied |= claimed
    for k in sorted(slots):
        if k not in occupied:
            r, c = k
            add_cell([k], [rows[r]["id"], cols[c]["id"]],
                     min(rows[r].get("score", 1), cols[c].get("score", 1)))
    cells.sort(key=lambda cell: (cell["row_start"], cell["col_start"]))
    for i, cell in enumerate(cells):
        cell["id"] = f"c{i}"
    return {"rows": rows, "columns": cols, "cells": cells,
            "diagnostics": diagnostics, "slot_count": len(slots)}
