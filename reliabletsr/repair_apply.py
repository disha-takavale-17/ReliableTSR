"""Deterministic repair application and geometric grid validation.

Applies proposed cell repairs to a TATR-predicted grid and enforces that the
resulting grid is strictly rectangular and non-overlapping. Any edit violating
preconditions, slot occupancy, or geometric consistency is safely rejected.
"""
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

from .geometry import area, intersection, union_box


class GridRepairError(ValueError):
    """Raised when a repair edit violates grid topology or geometry."""
    pass


def compute_slot_boxes(grid: Dict[str, Any]) -> Dict[Tuple[int, int], List[float]]:
    """Compute bounding box for every (row, col) intersection slot."""
    slots = {}
    rows = grid.get("rows", [])
    cols = grid.get("columns", [])
    for r_idx, r_obj in enumerate(rows):
        for c_idx, c_obj in enumerate(cols):
            box = intersection(r_obj["bbox"], c_obj["bbox"])
            if area(box) > 0:
                slots[(r_idx, c_idx)] = box
            else:
                # Fallback to combined bounding coordinates
                slots[(r_idx, c_idx)] = [
                    c_obj["bbox"][0],
                    r_obj["bbox"][1],
                    c_obj["bbox"][2],
                    r_obj["bbox"][3],
                ]
    return slots


def cell_slots(cell: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """Return set of (row, col) slot tuples occupied by a cell."""
    return {(r, c) for r in cell.get("rows", []) for c in cell.get("cols", [])}


def is_rectangular(cell: Dict[str, Any]) -> bool:
    """Verify that cell rows and columns form an unbroken rectangular matrix."""
    rows = cell.get("rows", [])
    cols = cell.get("cols", [])
    if not rows or not cols:
        return False
    r_start, r_end = cell.get("row_start", min(rows)), cell.get("row_end", max(rows))
    c_start, c_end = cell.get("col_start", min(cols)), cell.get("col_end", max(cols))

    if sorted(rows) != list(range(r_start, r_end + 1)):
        return False
    if sorted(cols) != list(range(c_start, c_end + 1)):
        return False

    return cell_slots(cell) == {(r, c) for r in range(r_start, r_end + 1) for c in range(c_start, c_end + 1)}


def validate_grid(
    grid: Dict[str, Any], slot_boxes: Dict[Tuple[int, int], List[float]]
) -> Tuple[bool, str]:
    """Validate that grid cells are strictly rectangular and non-overlapping."""
    cells = grid.get("cells", [])
    rows = grid.get("rows", [])
    cols = grid.get("columns", [])
    row_count = len(rows)
    col_count = len(cols)

    # 1. Check unique IDs
    ids = [c["id"] for c in cells]
    if len(set(ids)) != len(ids):
        return False, "Duplicate cell IDs detected in grid."

    # 2. Check bounds and rectangularity for each cell
    slot_occupancy: Dict[Tuple[int, int], str] = {}
    for cell in cells:
        cid = cell["id"]
        if not is_rectangular(cell):
            return False, f"Cell {cid} has a non-rectangular span."

        r_start, r_end = cell["row_start"], cell["row_end"]
        c_start, c_end = cell["col_start"], cell["col_end"]

        if not (0 <= r_start <= r_end < row_count):
            return False, f"Cell {cid} row indices [{r_start}, {r_end}] out of range [0, {row_count-1}]."
        if not (0 <= c_start <= c_end < col_count):
            return False, f"Cell {cid} col indices [{c_start}, {c_end}] out of range [0, {col_count-1}]."

        # 3. Check slot overlap
        for slot in cell_slots(cell):
            if slot in slot_occupancy:
                return False, f"Slot {slot} overlap between {slot_occupancy[slot]} and {cid}."
            slot_occupancy[slot] = cid

    # 4. Check geometric overlap between bounding boxes
    for i, c1 in enumerate(cells):
        for c2 in cells[i + 1 :]:
            box1, box2 = c1["bbox"], c2["bbox"]
            inter = area(intersection(box1, box2))
            min_area = min(area(box1), area(box2))
            if min_area > 0 and (inter / min_area) > 0.25:
                return False, f"Geometric box overlap ({inter/min_area:.1%}) between {c1['id']} and {c2['id']}."

    return True, ""


def apply_single_repair(
    grid: Dict[str, Any],
    proposal: Dict[str, Any],
    slot_boxes: Dict[Tuple[int, int], List[float]],
) -> Tuple[Dict[str, Any], bool, str]:
    """Apply one repair proposal to the grid.

    Returns:
        (updated_grid, accepted, reason)
    """
    action = proposal.get("action", "no_change")
    cid = proposal.get("cell_id", "")
    params = proposal.get("params", {}) or {}

    if action == "no_change":
        return grid, True, "No change requested."

    cells = {c["id"]: deepcopy(c) for c in grid.get("cells", [])}
    if cid not in cells:
        return grid, False, f"Target cell {cid} does not exist in grid."

    target = cells[cid]
    row_count = len(grid.get("rows", []))
    col_count = len(grid.get("columns", []))

    if action == "set_header":
        target["header"] = bool(params.get("header", True))
        grid_copy = deepcopy(grid)
        grid_copy["cells"] = list(cells.values())
        return grid_copy, True, f"Header updated for {cid}."

    elif action == "delete_cell":
        del cells[cid]
        grid_copy = deepcopy(grid)
        grid_copy["cells"] = list(cells.values())
        valid, msg = validate_grid(grid_copy, slot_boxes)
        if not valid:
            return grid, False, f"Delete cell rejected: {msg}"
        return grid_copy, True, f"Cell {cid} deleted."

    elif action == "set_span":
        r0 = params.get("row_start", target["row_start"])
        r1 = params.get("row_end", target["row_end"])
        c0 = params.get("col_start", target["col_start"])
        c1 = params.get("col_end", target["col_end"])

        if not (0 <= r0 <= r1 < row_count and 0 <= c0 <= c1 < col_count):
            return grid, False, f"set_span coordinates [{r0},{r1},{c0},{c1}] out of bounds."

        new_slots = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        # Check collision with other cells
        for other_id, other_cell in cells.items():
            if other_id == cid:
                continue
            if cell_slots(other_cell) & new_slots:
                return grid, False, f"set_span for {cid} collides with slots of {other_id}."

        target["row_start"] = r0
        target["row_end"] = r1
        target["rows"] = list(range(r0, r1 + 1))
        target["col_start"] = c0
        target["col_end"] = c1
        target["cols"] = list(range(c0, c1 + 1))

        # Recompute box from slots
        boxes = [slot_boxes[s] for s in new_slots if s in slot_boxes]
        if boxes:
            target["bbox"] = union_box(boxes)

        grid_copy = deepcopy(grid)
        grid_copy["cells"] = list(cells.values())
        valid, msg = validate_grid(grid_copy, slot_boxes)
        if not valid:
            return grid, False, f"set_span rejected: {msg}"
        return grid_copy, True, f"Span updated for {cid}."

    elif action == "merge":
        merge_with = params.get("merge_with", [])
        if not isinstance(merge_with, list) or not merge_with:
            return grid, False, "merge action requires a nonempty 'merge_with' list of cell IDs."

        all_ids = [cid] + [m for m in merge_with if m != cid]
        for mid in all_ids:
            if mid not in cells:
                return grid, False, f"Cannot merge nonexistent cell {mid}."

        to_merge = [cells[mid] for mid in all_ids]
        combined_slots = set().union(*(cell_slots(c) for c in to_merge))

        # Check if combined slots form an exact unbroken rectangle
        r_vals = {r for r, _ in combined_slots}
        c_vals = {c for _, c in combined_slots}
        r_min, r_max = min(r_vals), max(r_vals)
        c_min, c_max = min(c_vals), max(c_vals)
        expected_slots = {(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)}

        if combined_slots != expected_slots:
            return grid, False, "Merged cells do not form a single rectangular region."

        # Check collision with other cells
        other_cells = {k: v for k, v in cells.items() if k not in all_ids}
        for other_id, other_cell in other_cells.items():
            if cell_slots(other_cell) & combined_slots:
                return grid, False, f"Merged cell region collides with slots of {other_id}."

        # Create merged cell
        boxes = [c["bbox"] for c in to_merge]
        merged_cell = {
            "id": cid,
            "bbox": union_box(boxes),
            "rows": list(range(r_min, r_max + 1)),
            "cols": list(range(c_min, c_max + 1)),
            "row_start": r_min,
            "row_end": r_max,
            "col_start": c_min,
            "col_end": c_max,
            "header": any(c.get("header", False) for c in to_merge),
            "projected_row_header": any(c.get("projected_row_header", False) for c in to_merge),
            "score": min(c.get("score", 1.0) for c in to_merge),
            "source": f"merged({','.join(all_ids)})",
        }

        del cells[cid]
        for mid in merge_with:
            if mid in cells:
                del cells[mid]
        cells[cid] = merged_cell

        grid_copy = deepcopy(grid)
        grid_copy["cells"] = list(cells.values())
        valid, msg = validate_grid(grid_copy, slot_boxes)
        if not valid:
            return grid, False, f"Merge rejected: {msg}"
        return grid_copy, True, f"Merged {all_ids} into {cid}."

    elif action == "split":
        slots = cell_slots(target)
        if len(slots) <= 1:
            return grid, False, f"Cannot split cell {cid}: occupies only 1 slot (fixed grid resolution)."

        direction = str(params.get("direction", "vertical")).lower()
        r_start, r_end = target["row_start"], target["row_end"]
        c_start, c_end = target["col_start"], target["col_end"]

        split_pieces: List[Dict[str, Any]] = []

        if direction == "horizontal" and (r_end > r_start):
            # Split across rows
            mid_r = (r_start + r_end) // 2
            ranges = [(r_start, mid_r, c_start, c_end), (mid_r + 1, r_end, c_start, c_end)]
        elif direction == "vertical" and (c_end > c_start):
            # Split across cols
            mid_c = (c_start + c_end) // 2
            ranges = [(r_start, r_end, c_start, mid_c), (r_start, r_end, mid_c + 1, c_end)]
        elif (c_end > c_start):
            mid_c = (c_start + c_end) // 2
            ranges = [(r_start, r_end, c_start, mid_c), (r_start, r_end, mid_c + 1, c_end)]
        elif (r_end > r_start):
            mid_r = (r_start + r_end) // 2
            ranges = [(r_start, mid_r, c_start, c_end), (mid_r + 1, r_end, c_start, c_end)]
        else:
            return grid, False, f"Cannot split cell {cid} in direction '{direction}'."

        for idx, (r0, r1, c0, c1) in enumerate(ranges):
            piece_slots = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
            piece_boxes = [slot_boxes[s] for s in piece_slots if s in slot_boxes]
            box = union_box(piece_boxes) if piece_boxes else target["bbox"]
            new_cid = f"{cid}_p{idx+1}"
            split_pieces.append({
                "id": new_cid,
                "bbox": box,
                "rows": list(range(r0, r1 + 1)),
                "cols": list(range(c0, c1 + 1)),
                "row_start": r0,
                "row_end": r1,
                "col_start": c0,
                "col_end": c1,
                "header": target.get("header", False),
                "projected_row_header": target.get("projected_row_header", False),
                "score": target.get("score", 1.0),
                "source": f"split({cid})",
            })

        del cells[cid]
        for p in split_pieces:
            cells[p["id"]] = p

        grid_copy = deepcopy(grid)
        grid_copy["cells"] = list(cells.values())
        valid, msg = validate_grid(grid_copy, slot_boxes)
        if not valid:
            return grid, False, f"Split rejected: {msg}"
        return grid_copy, True, f"Split cell {cid} into {[p['id'] for p in split_pieces]}."

    return grid, False, f"Unsupported action: {action}"


def apply_repairs(
    grid: Dict[str, Any], proposals: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Sequentially apply repair proposals with rollback on validation failure.

    Returns:
        (final_grid, execution_log)
    """
    current_grid = deepcopy(grid)
    slot_boxes = compute_slot_boxes(current_grid)
    execution_log: List[Dict[str, Any]] = []

    for proposal in proposals:
        cid = proposal.get("cell_id", "")
        action = proposal.get("action", "no_change")

        if action == "no_change":
            execution_log.append({
                "cell_id": cid,
                "action": action,
                "accepted": True,
                "reason": "No change.",
            })
            continue

        updated_grid, accepted, reason = apply_single_repair(current_grid, proposal, slot_boxes)
        if accepted:
            current_grid = updated_grid
            execution_log.append({
                "cell_id": cid,
                "action": action,
                "accepted": True,
                "reason": reason,
            })
        else:
            execution_log.append({
                "cell_id": cid,
                "action": action,
                "accepted": False,
                "reason": reason,
            })

    # Sort final cells by row_start, col_start
    current_grid["cells"] = sorted(
        current_grid.get("cells", []),
        key=lambda c: (c.get("row_start", 0), c.get("col_start", 0), c.get("id", "")),
    )
    return current_grid, execution_log
