"""Prompt engineering templates for targeted spatial table auditing and extraction."""
from typing import Dict, List


def build_audit_prompt(risk_data: Dict) -> str:
    """Format spatial risk findings into a forensic auditing prompt for Qwen2.5-VL."""
    reasons: List[str] = []
    for r in risk_data.get("reasons", []):
        if r.get("severity", 0) >= 0.05:
            sig = r.get("signal", "").replace("_", " ").title()
            reason = r.get("reason", "")
            cells = ", ".join(r.get("cells", [])) or "table-level"
            reasons.append(f"- **{sig}**: {reason} (Affected cells: {cells})")

    findings_text = "\n".join(reasons) if reasons else "No specific high-severity signals flagged."

    return (
        "You are an expert Document AI Table Auditor. "
        "A geometric object detection model (TATR) scanned this table and flagged the following potential anomalies:\n\n"
        f"{findings_text}\n\n"
        "Please visually inspect the table image carefully and provide an audit:\n"
        "1. **Verdict**: For each flagged issue, state clearly if it is a REAL ERROR (e.g. split cell, merged column, missed header) "
        "or a FALSE ALARM (e.g. valid blank space, stylistic line spacing, multiline cell).\n"
        "2. **Layout Explanation**: Describe the true layout (number of columns, headers, spanning cells).\n"
        "3. **Recommendation**: State whether the table is safe to use in automated pipelines."
    )


def build_extraction_prompt(format_type: str = "markdown") -> str:
    """Build prompt for transcribing table image directly to clean tabular format."""
    return (
        "You are an expert table extraction system. "
        "Transcribe the entire table from this image into a clean, well-structured GitHub-flavored Markdown table.\n"
        "Rules:\n"
        "- Preserve all header hierarchies and row alignments.\n"
        "- Retain all numerical values, punctuation, and currencies accurately.\n"
        "- For empty cells, leave them blank between pipe symbols (`| |`).\n"
        "- Do NOT add conversational preamble or summary. Output ONLY the raw Markdown table."
    )
