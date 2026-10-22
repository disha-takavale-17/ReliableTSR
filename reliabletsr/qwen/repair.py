"""Targeted spatial table repair using Qwen2.5-VL with visual bounding-box grounding."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

from .client import QwenClient

VALID_VERDICTS = {"REAL_ERROR", "FALSE_ALARM"}
VALID_ACTIONS = {"no_change", "merge", "split", "set_span", "set_header", "delete_cell"}
EXCLUDED_SIGNALS = {"suspicious_span", "empty_grid"}


def get_eligible_flagged_cells(
    prediction: Dict[str, Any], risk_data: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], bool]:
    """Extract flagged cells, excluding suspicious_span and empty_grid flags.

    Returns:
        flagged_cells: List of cell dicts from prediction
        cell_signals: Mapping from cell_id to list of risk signals that flagged it
        has_excluded_flags: True if table contains suspicious_span or empty_grid flags
    """
    signals_present = {f.get("signal", "") for f in risk_data.get("reasons", [])}
    has_excluded_flags = bool(signals_present & EXCLUDED_SIGNALS)

    cell_signals: Dict[str, List[str]] = {}
    for f in risk_data.get("reasons", []):
        sig = f.get("signal", "")
        if sig in EXCLUDED_SIGNALS:
            continue
        for cid in f.get("cells", []):
            if cid:
                cell_signals.setdefault(cid, []).append(sig)

    cells_by_id = {c["id"]: c for c in prediction.get("cells", [])}
    flagged = []
    for cid in sorted(cell_signals.keys()):
        if cid in cells_by_id:
            flagged.append(cells_by_id[cid])

    return flagged, cell_signals, has_excluded_flags


def draw_flagged_cells(
    image: Image.Image, flagged_cells: List[Dict[str, Any]]
) -> Image.Image:
    """Draw numbered bounding boxes and labels for flagged cells on a copy of the table crop."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    palette = [
        ("#D90429", "#FFFFFF"),  # Red with white text
        ("#0077B6", "#FFFFFF"),  # Blue with white text
        ("#2A9D8F", "#FFFFFF"),  # Teal with white text
        ("#E76F51", "#FFFFFF"),  # Coral with white text
        ("#7209B7", "#FFFFFF"),  # Purple with white text
    ]

    for idx, cell in enumerate(flagged_cells, 1):
        box = cell["bbox"]
        xmin, ymin, xmax, ymax = box
        outline_color, text_color = palette[(idx - 1) % len(palette)]

        # Draw cell bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=outline_color, width=3)

        # Draw index badge at top-left
        label = f"#{idx}:{cell['id']}"
        text_bbox = draw.textbbox((xmin, ymin), label, font=font)
        badge_w = text_bbox[2] - text_bbox[0] + 6
        badge_h = text_bbox[3] - text_bbox[1] + 4
        badge_y0 = max(0, ymin - badge_h)
        badge_x0 = max(0, xmin)

        draw.rectangle(
            [badge_x0, badge_y0, badge_x0 + badge_w, badge_y0 + badge_h],
            fill=outline_color,
        )
        draw.text((badge_x0 + 3, badge_y0 + 2), label, fill=text_color, font=font)

    return annotated


def build_repair_prompt(
    flagged_cells: List[Dict[str, Any]],
    cell_signals: Dict[str, List[str]],
    image_size: Tuple[int, int],
) -> str:
    """Build grounded multimodal repair prompt with cell coordinate and signal specifications."""
    cell_descriptions: List[str] = []
    for idx, cell in enumerate(flagged_cells, 1):
        cid = cell["id"]
        signals = ", ".join(cell_signals.get(cid, ["unspecified"]))
        box = [round(float(v), 1) for v in cell["bbox"]]
        rows = cell.get("rows", [cell.get("row_start", 0)])
        cols = cell.get("cols", [cell.get("col_start", 0)])
        header = cell.get("header", False)
        desc = (
            f"- Box #{idx} ({cid}): BoundingBox={box}, Rows={rows}, Cols={cols}, "
            f"Header={header}, Flagged by: [{signals}]"
        )
        cell_descriptions.append(desc)

    cells_text = "\n".join(cell_descriptions)

    return (
        "You are an expert Document AI Table Auditor and Structural Repair Specialist.\n"
        "A Table Transformer (TATR) scanned this table, and geometric risk checks flagged the following "
        "potential cell anomalies (highlighted with numbered colored boxes on the image):\n\n"
        f"{cells_text}\n\n"
        "Table dimensions: width={image_size[0]}px, height={image_size[1]}px.\n\n"
        "Please visually inspect each highlighted cell against the image and determine whether each is a "
        "REAL_ERROR or a FALSE_ALARM, and provide a repair action.\n\n"
        "Output ONLY a raw JSON array containing EXACTLY ONE object per highlighted cell with this schema:\n"
        "[\n"
        "  {\n"
        '    "cell_id": "<cell_id>",\n'
        '    "verdict": "REAL_ERROR" or "FALSE_ALARM",\n'
        '    "action": "no_change" | "merge" | "split" | "set_span" | "set_header" | "delete_cell",\n'
        '    "params": {}\n'
        "  }\n"
        "]\n\n"
        "Action rules:\n"
        '- "no_change": The cell is structurally acceptable. params: {}\n'
        '- "merge": Merge this cell with another cell. params: {"merge_with": ["<other_cell_id>"]}\n'
        '- "split": Cell encompasses multiple table entries. params: {"direction": "horizontal"|"vertical"}\n'
        '- "set_span": Fix cell row/column coordinates. params: {"row_start": int, "row_end": int, "col_start": int, "col_end": int}\n'
        '- "set_header": Change header status. params: {"header": true|false}\n'
        '- "delete_cell": Cell is a spurious false-positive detection. params: {}\n\n'
        "Strict Requirements:\n"
        "1. verdict must be exactly 'REAL_ERROR' or 'FALSE_ALARM'.\n"
        "2. If verdict is 'FALSE_ALARM', action must be 'no_change'.\n"
        "3. Every flagged cell must have an entry in the JSON array.\n"
        "4. Output raw JSON ONLY. No markdown conversational text or explanation."
    )


def validate_repair_response(
    raw_text: str, flagged_cell_ids: List[str]
) -> Tuple[List[Dict[str, Any]], int]:
    """Parse and validate Qwen repair JSON response.

    Invalid entries or missing cells are reverted to no_change and counted.
    Returns:
        proposals: List of valid repair operation dicts
        invalid_count: Count of invalid or malformed proposals
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()

    parsed_items: List[Any] = []
    invalid_count = 0

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            parsed_items = data
        elif isinstance(data, dict):
            # If wrapped in a dictionary like {"repairs": [...]} or {"cells": [...]}
            for key in ("repairs", "cells", "proposals", "results"):
                if key in data and isinstance(data[key], list):
                    parsed_items = data[key]
                    break
            if not parsed_items:
                parsed_items = [data]
        else:
            invalid_count += len(flagged_cell_ids)
    except Exception:
        # Malformed JSON
        return [
            {
                "cell_id": cid,
                "verdict": "FALSE_ALARM",
                "action": "no_change",
                "params": {},
                "valid": False,
                "error": "Malformed JSON",
            }
            for cid in flagged_cell_ids
        ], len(flagged_cell_ids)

    # Index parsed proposals by cell_id
    proposals_by_id: Dict[str, Dict[str, Any]] = {}
    for item in parsed_items:
        if not isinstance(item, dict):
            invalid_count += 1
            continue
        cid = str(item.get("cell_id", "")).strip()
        if not cid or cid not in flagged_cell_ids:
            invalid_count += 1
            continue

        verdict = str(item.get("verdict", "")).strip().upper()
        action = str(item.get("action", "")).strip().lower()
        params = item.get("params", {})

        is_valid = True
        err_msg = ""

        if verdict not in VALID_VERDICTS:
            is_valid = False
            err_msg = f"Invalid verdict: {verdict}"
        elif action not in VALID_ACTIONS:
            is_valid = False
            err_msg = f"Invalid action: {action}"
        elif not isinstance(params, dict):
            is_valid = False
            err_msg = "Params must be a dictionary"

        if is_valid:
            proposals_by_id[cid] = {
                "cell_id": cid,
                "verdict": verdict,
                "action": action,
                "params": params,
                "valid": True,
            }
        else:
            invalid_count += 1
            proposals_by_id[cid] = {
                "cell_id": cid,
                "verdict": "FALSE_ALARM",
                "action": "no_change",
                "params": {},
                "valid": False,
                "error": err_msg,
            }

    # Fill in any missing flagged cell
    final_proposals: List[Dict[str, Any]] = []
    for cid in flagged_cell_ids:
        if cid in proposals_by_id:
            final_proposals.append(proposals_by_id[cid])
        else:
            invalid_count += 1
            final_proposals.append(
                {
                    "cell_id": cid,
                    "verdict": "FALSE_ALARM",
                    "action": "no_change",
                    "params": {},
                    "valid": False,
                    "error": "Cell missing from model output",
                }
            )

    return final_proposals, invalid_count


class QwenRepairAuditor:
    """Manages Qwen repair generation, caching, and proposal validation."""

    def __init__(
        self,
        client: Optional[QwenClient] = None,
        cache_dir: str = "data/repair_cache",
    ):
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, table_id: str, prompt: str) -> Path:
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in table_id)
        return self.cache_dir / f"{safe_id}_{prompt_hash}.json"

    def audit_and_repair(
        self,
        image: Image.Image,
        prediction: Dict[str, Any],
        risk_data: Dict[str, Any],
        table_id: str = "table_1",
    ) -> Dict[str, Any]:
        """Run grounded Qwen audit and repair proposal for flagged cells."""
        flagged_cells, cell_signals, excluded_only = get_eligible_flagged_cells(
            prediction, risk_data
        )

        if not flagged_cells:
            return {
                "table_id": table_id,
                "eligible": False,
                "excluded_only": excluded_only,
                "flagged_cells": [],
                "proposals": [],
                "invalid_count": 0,
                "qwen_called": False,
                "raw_response": "",
            }

        flagged_ids = [c["id"] for c in flagged_cells]
        annotated_image = draw_flagged_cells(image, flagged_cells)
        prompt = build_repair_prompt(
            flagged_cells, cell_signals, (image.width, image.height)
        )

        cache_file = self._cache_path(table_id, prompt)
        qwen_called = False
        raw_response = ""

        if cache_file.exists():
            try:
                cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                raw_response = cached_data.get("raw_response", "")
            except Exception:
                raw_response = ""

        if not raw_response:
            if self.client is None:
                api_key = os.getenv("OPENROUTER_API_KEY", "")
                self.client = QwenClient(api_key=api_key)

            raw_response = self.client.query(annotated_image, prompt)
            qwen_called = True
            cache_file.write_text(
                json.dumps(
                    {
                        "table_id": table_id,
                        "prompt": prompt,
                        "raw_response": raw_response,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        proposals, invalid_count = validate_repair_response(raw_response, flagged_ids)

        return {
            "table_id": table_id,
            "eligible": True,
            "excluded_only": excluded_only,
            "flagged_cells": flagged_ids,
            "proposals": proposals,
            "invalid_count": invalid_count,
            "qwen_called": qwen_called,
            "raw_response": raw_response,
        }
