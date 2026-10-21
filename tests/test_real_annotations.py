from pathlib import Path

import pytest

from reliabletsr.structure import build_grid, parse_xml


def test_selected_official_xmls_convert_to_nonempty_grids():
    xmls = sorted(Path("data/pairs/development/xml").glob("*.xml"))
    if not xmls:
        pytest.skip("Run prepare to enable real annotation checks")
    for path in xmls[:10]:
        raw = parse_xml(path)
        grid = build_grid(raw["objects"], threshold=0)
        assert raw["size"] and grid["rows"] and grid["columns"] and grid["cells"]
        assert all(c["rows"] and c["cols"] for c in grid["cells"])
