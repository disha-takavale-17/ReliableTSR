"""Unit tests for Qwen grounded repair prompt builder and JSON validation."""
from PIL import Image

from reliabletsr.qwen.repair import (
    build_repair_prompt,
    draw_flagged_cells,
    get_eligible_flagged_cells,
    validate_repair_response,
)


def test_get_eligible_flagged_cells():
    prediction = {
        "cells": [
            {"id": "c0", "bbox": [10, 10, 50, 50], "rows": [0], "cols": [0], "header": True},
            {"id": "c1", "bbox": [50, 10, 100, 50], "rows": [0], "cols": [1], "header": True},
            {"id": "c2", "bbox": [10, 50, 50, 100], "rows": [1], "cols": [0], "header": False},
        ]
    }
    risk_data = {
        "reasons": [
            {"signal": "cell_overlap", "cells": ["c0", "c1"]},
            {"signal": "suspicious_span", "cells": []},
        ]
    }
    flagged, signals, has_excluded_flags = get_eligible_flagged_cells(prediction, risk_data)
    assert len(flagged) == 2
    assert [c["id"] for c in flagged] == ["c0", "c1"]
    assert signals["c0"] == ["cell_overlap"]
    assert has_excluded_flags is True


def test_draw_flagged_cells():
    image = Image.new("RGB", (200, 200), color="white")
    flagged = [
        {"id": "c0", "bbox": [10, 10, 50, 50]},
        {"id": "c1", "bbox": [60, 10, 120, 50]},
    ]
    annotated = draw_flagged_cells(image, flagged)
    assert annotated.size == (200, 200)
    # Check that image was drawn on (not completely white)
    assert annotated.tobytes() != image.tobytes()


def test_build_repair_prompt():
    flagged = [
        {"id": "c0", "bbox": [10, 10, 50, 50], "rows": [0], "cols": [0], "header": True}
    ]
    signals = {"c0": ["low_confidence"]}
    prompt = build_repair_prompt(flagged, signals, (200, 200))
    assert "c0" in prompt
    assert "low_confidence" in prompt
    assert "REAL_ERROR" in prompt
    assert "FALSE_ALARM" in prompt


def test_validate_repair_response_valid():
    raw_json = """
    ```json
    [
      {
        "cell_id": "c0",
        "verdict": "REAL_ERROR",
        "action": "merge",
        "params": {"merge_with": ["c1"]}
      },
      {
        "cell_id": "c1",
        "verdict": "FALSE_ALARM",
        "action": "no_change",
        "params": {}
      }
    ]
    ```
    """
    proposals, invalid_count = validate_repair_response(raw_json, ["c0", "c1"])
    assert invalid_count == 0
    assert len(proposals) == 2
    assert proposals[0]["action"] == "merge"
    assert proposals[1]["action"] == "no_change"


def test_validate_repair_response_invalid_fallback():
    raw_text = "I think the table looks fine. Here is an invalid action: ```json [{\"cell_id\": \"c0\", \"verdict\": \"UNKNOWN\", \"action\": \"magic_wand\"}] ```"
    proposals, invalid_count = validate_repair_response(raw_text, ["c0", "c1"])
    assert invalid_count >= 1
    assert len(proposals) == 2
    # c0 had invalid action, c1 was missing -> both should fall back to no_change
    assert proposals[0]["action"] == "no_change"
    assert proposals[1]["action"] == "no_change"
