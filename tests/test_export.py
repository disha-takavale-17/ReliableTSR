"""Unit tests for Markdown table parsing and export generation."""
from reliabletsr.export.excel_exporter import TableExporter


def test_markdown_parsing():
    sample_md = """
    | Product | Q1 Sales | Q2 Sales |
    | :--- | :---: | ---: |
    | Widget A | $1,200 | $1,500 |
    | Widget B | $850 | $920 |
    """
    rows = TableExporter.parse_markdown_table(sample_md)
    assert len(rows) == 3
    assert rows[0] == ["Product", "Q1 Sales", "Q2 Sales"]
    assert rows[1] == ["Widget A", "$1,200", "$1,500"]
    assert rows[2] == ["Widget B", "$850", "$920"]


def test_csv_export():
    sample_md = """
    | Item | Price |
    | --- | --- |
    | Book | 15.00 |
    """
    csv_bytes = TableExporter.to_csv_bytes(sample_md)
    assert b"Item,Price" in csv_bytes
    assert b"Book,15.00" in csv_bytes


def test_excel_or_fallback_export():
    sample_md = """
    | Header 1 | Header 2 |
    | --- | --- |
    | Val 1 | Val 2 |
    """
    content_bytes, mime = TableExporter.to_excel_bytes(sample_md)
    assert len(content_bytes) > 0
    assert mime in (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/csv",
    )
