"""Unit tests for deterministic repair application and grid validation."""
import pytest
from reliabletsr.repair_apply import apply_repairs, apply_single_repair, compute_slot_boxes, validate_grid


@pytest.fixture
def base_2x2_grid():
    rows = [
        {"id": "r0", "bbox": [0, 0, 100, 50]},
        {"id": "r1", "bbox": [0, 50, 100, 100]},
    ]
    cols = [
        {"id": "c0_col", "bbox": [0, 0, 50, 100]},
        {"id": "c1_col", "bbox": [50, 0, 100, 100]},
    ]
    cells = [
        {
            "id": "c0",
            "bbox": [0, 0, 50, 50],
            "rows": [0],
            "cols": [0],
            "row_start": 0,
            "row_end": 0,
            "col_start": 0,
            "col_end": 0,
            "header": True,
            "projected_row_header": False,
            "score": 0.99,
            "source": "tatr",
        },
        {
            "id": "c1",
            "bbox": [50, 0, 100, 50],
            "rows": [0],
            "cols": [1],
            "row_start": 0,
            "row_end": 0,
            "col_start": 1,
            "col_end": 1,
            "header": True,
            "projected_row_header": False,
            "score": 0.99,
            "source": "tatr",
        },
        {
            "id": "c2",
            "bbox": [0, 50, 50, 100],
            "rows": [1],
            "cols": [0],
            "row_start": 1,
            "row_end": 1,
            "col_start": 0,
            "col_end": 0,
            "header": False,
            "projected_row_header": False,
            "score": 0.99,
            "source": "tatr",
        },
        {
            "id": "c3",
            "bbox": [50, 50, 100, 100],
            "rows": [1],
            "cols": [1],
            "row_start": 1,
            "row_end": 1,
            "col_start": 1,
            "col_end": 1,
            "header": False,
            "projected_row_header": False,
            "score": 0.99,
            "source": "tatr",
        },
    ]
    return {"rows": rows, "columns": cols, "cells": cells}


def test_no_change(base_2x2_grid):
    proposal = [{"cell_id": "c0", "action": "no_change", "params": {}}]
    grid, log = apply_repairs(base_2x2_grid, proposal)
    assert len(grid["cells"]) == 4
    assert log[0]["accepted"] is True


def test_set_header(base_2x2_grid):
    proposal = [{"cell_id": "c2", "action": "set_header", "params": {"header": True}}]
    grid, log = apply_repairs(base_2x2_grid, proposal)
    assert log[0]["accepted"] is True
    c2 = next(c for c in grid["cells"] if c["id"] == "c2")
    assert c2["header"] is True


def test_merge_adjacent_cells(base_2x2_grid):
    # Merge c0 and c1 (top row)
    proposal = [{"cell_id": "c0", "action": "merge", "params": {"merge_with": ["c1"]}}]
    grid, log = apply_repairs(base_2x2_grid, proposal)
    assert log[0]["accepted"] is True
    assert len(grid["cells"]) == 3
    c0 = next(c for c in grid["cells"] if c["id"] == "c0")
    assert c0["cols"] == [0, 1]
    assert c0["col_start"] == 0
    assert c0["col_end"] == 1
    assert "c1" not in [c["id"] for c in grid["cells"]]


def test_merge_diagonal_rejects(base_2x2_grid):
    # Merge c0 and c3 (diagonal: non-rectangular union)
    proposal = [{"cell_id": "c0", "action": "merge", "params": {"merge_with": ["c3"]}}]
    grid, log = apply_repairs(base_2x2_grid, proposal)
    assert log[0]["accepted"] is False
    assert len(grid["cells"]) == 4  # Unaltered


def test_split_spanning_cell(base_2x2_grid):
    # First merge c0 and c1 to form a span
    proposal1 = [{"cell_id": "c0", "action": "merge", "params": {"merge_with": ["c1"]}}]
    grid1, _ = apply_repairs(base_2x2_grid, proposal1)

    # Now split c0 vertically
    proposal2 = [{"cell_id": "c0", "action": "split", "params": {"direction": "vertical"}}]
    grid2, log2 = apply_repairs(grid1, proposal2)
    assert log2[0]["accepted"] is True
    assert len(grid2["cells"]) == 4
    assert any(c["id"] == "c0_p1" for c in grid2["cells"])
    assert any(c["id"] == "c0_p2" for c in grid2["cells"])


def test_split_1x1_cell_rejects(base_2x2_grid):
    proposal = [{"cell_id": "c0", "action": "split", "params": {"direction": "vertical"}}]
    grid, log = apply_repairs(base_2x2_grid, proposal)
    assert log[0]["accepted"] is False
    assert len(grid["cells"]) == 4


def test_set_span_collision_rejects(base_2x2_grid):
    # Try to expand c0 to span cols [0, 1] without merging, colliding with c1
    proposal = [
        {
            "cell_id": "c0",
            "action": "set_span",
            "params": {"row_start": 0, "row_end": 0, "col_start": 0, "col_end": 1},
        }
    ]
    grid, log = apply_repairs(base_2x2_grid, proposal)
    assert log[0]["accepted"] is False
    assert len(grid["cells"]) == 4


def test_delete_cell(base_2x2_grid):
    proposal = [{"cell_id": "c3", "action": "delete_cell", "params": {}}]
    grid, log = apply_repairs(base_2x2_grid, proposal)
    assert log[0]["accepted"] is True
    assert len(grid["cells"]) == 3
    assert "c3" not in [c["id"] for c in grid["cells"]]
