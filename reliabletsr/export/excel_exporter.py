"""In-memory exporters for converting Markdown tables into Excel (.xlsx) and CSV bytes."""
import csv
import io
import re
from typing import List, Tuple


class TableExporter:
    """Parses Markdown tables into structured rows and exports to Excel or CSV bytes."""

    @staticmethod
    def parse_table(raw_text: str) -> List[List[str]]:
        """Extract structured matrix of cells from Markdown, CSV, TSV, or wrapped code blocks."""
        if not raw_text or not isinstance(raw_text, str):
            return []

        text = raw_text.strip()
        # Strip markdown code fences if wrapped (e.g. ```csv ... ``` or ```markdown ... ```)
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        # Check if predominantly pipe-delimited (| col1 | col2 |)
        pipe_lines = [l for l in lines if l.startswith("|") or l.endswith("|") or "|" in l]
        if len(pipe_lines) >= len(lines) * 0.5 and "|" in lines[0]:
            rows: List[List[str]] = []
            for line in lines:
                if not line.startswith("|") and not line.endswith("|") and "|" not in line:
                    continue
                if re.match(r"^\|(\s*:?-+:?\s*\|)+$", line):
                    continue
                parts = line.split("|")
                if parts and parts[0] == "":
                    parts = parts[1:]
                if parts and parts[-1] == "":
                    parts = parts[:-1]
                cells = [c.strip() for c in parts]
                if any(cells):
                    rows.append(cells)
            if rows:
                return rows

        # Parse as CSV, TSV, or semicolon-delimited
        delimiter = ","
        first_line = lines[0]
        if "\t" in first_line and "," not in first_line:
            delimiter = "\t"
        elif ";" in first_line and "," not in first_line:
            delimiter = ";"

        try:
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            rows = [row for row in reader if any(cell.strip() for cell in row)]
            if rows:
                return rows
        except Exception:
            pass

        # Fallback: line-by-line whitespace or comma splitting
        return [[c.strip() for c in line.split(",") if c.strip()] for line in lines]

    # Backward compatibility alias
    parse_markdown_table = parse_table

    @classmethod
    def to_csv_bytes(cls, raw_text: str) -> bytes:
        """Convert extracted table text into normalized CSV byte stream."""
        if not raw_text:
            return b""
        rows = cls.parse_table(raw_text)
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        for row in rows:
            writer.writerow(row)
        return buf.getvalue().encode("utf-8")

    @classmethod
    def to_excel_bytes(cls, raw_text: str, sheet_name: str = "Extracted Table") -> Tuple[bytes, str]:
        """Convert extracted table text into Excel (.xlsx) bytes in memory.

        Returns:
            Tuple of (file_bytes, mime_type)
        """
        if not raw_text:
            return b"", "text/csv"
        rows = cls.parse_table(raw_text)
        if not rows:
            return b"", "text/csv"

        # Attempt to export via openpyxl
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = sheet_name

            header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
            regular_font = Font(name="Segoe UI", size=10)
            center_align = Alignment(horizontal="center", vertical="center")
            thin_border = Border(
                left=Side(style="thin", color="D9D9D9"),
                right=Side(style="thin", color="D9D9D9"),
                top=Side(style="thin", color="D9D9D9"),
                bottom=Side(style="thin", color="D9D9D9"),
            )

            for r_idx, row in enumerate(rows, start=1):
                for c_idx, val in enumerate(row, start=1):
                    cell = ws.cell(row=r_idx, column=c_idx, value=val)
                    cell.border = thin_border
                    if r_idx == 1:
                        cell.font = header_font
                        cell.fill = header_fill
                        cell.alignment = center_align
                    else:
                        cell.font = regular_font

            # Auto-adjust column widths
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                col_letter = openpyxl.utils.get_column_letter(col[0].column)
                ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

            out_buf = io.BytesIO()
            wb.save(out_buf)
            return out_buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        except ImportError:
            # Fallback to UTF-8 CSV if openpyxl is not installed
            return cls.to_csv_bytes(markdown_text), "text/csv"
