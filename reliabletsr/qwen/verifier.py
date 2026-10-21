"""High-level auditor and table extraction orchestrator using Qwen2.5-VL."""
from dataclasses import dataclass
from typing import Dict, Optional
from PIL import Image

from .client import QwenClient
from .prompts import build_audit_prompt, build_extraction_prompt


@dataclass
class AuditResult:
    table_id: str
    risk_score: float
    possible_error: bool
    qwen_verdict: str
    markdown_content: Optional[str] = None


class QwenTableAuditor:
    """Combines TATR heuristic risk data with Qwen multimodal verification."""

    def __init__(self, client: Optional[QwenClient] = None):
        self.client = client or QwenClient()

    def audit_table(self, image: Image.Image, risk_data: Dict, table_id: str = "table_1") -> AuditResult:
        """Audit flagged anomalies on a table crop."""
        score = risk_data.get("score", 0.0)
        possible_error = risk_data.get("possible_error", False)

        prompt = build_audit_prompt(risk_data)
        verdict = self.client.query(image, prompt)

        return AuditResult(
            table_id=table_id,
            risk_score=score,
            possible_error=possible_error,
            qwen_verdict=verdict,
        )

    def extract_table(self, image: Image.Image) -> str:
        """Transcribe table directly into clean Markdown table string."""
        prompt = build_extraction_prompt(format_type="markdown")
        return self.client.query(image, prompt)
