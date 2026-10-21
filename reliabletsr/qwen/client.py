"""Unified client for Alibaba Qwen2.5-VL via free cloud API or local Ollama."""
import base64
import io
import os
from typing import Optional
from PIL import Image
import requests

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
FREE_OPENROUTER_MODEL = "nvidia/nemotron-3.5-super-vl:free"
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "qwen2.5-vl:3b"


class QwenClient:
    """Dispatches multimodal vision-language queries to Qwen2.5-VL."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "openrouter",
        custom_endpoint: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: int = 60,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or ""
        self.provider = provider.lower()
        self.endpoint = custom_endpoint or (
            OLLAMA_ENDPOINT if self.provider == "ollama" else OPENROUTER_ENDPOINT
        )
        self.model = model or (
            DEFAULT_OLLAMA_MODEL if self.provider == "ollama" else FREE_OPENROUTER_MODEL
        )
        self.timeout = timeout_seconds

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to JPEG base64 string directly in memory."""
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def query(self, image: Image.Image, prompt: str) -> str:
        """Send image and prompt to Qwen, returning text response."""
        b64_image = self._image_to_base64(image)

        if self.provider == "ollama":
            return self._query_ollama(b64_image, prompt)
        else:
            return self._query_openrouter(b64_image, prompt)

    def _query_openrouter(self, b64_image: str, prompt: str) -> str:
        """Invoke OpenRouter free endpoint."""
        clean_key = (self.api_key or "").strip().strip("'\"")
        if clean_key.lower().startswith("bearer "):
            clean_key = clean_key[7:].strip()

        if not clean_key or len(clean_key) < 10:
            raise ValueError(
                "Invalid or missing OpenRouter API Key. "
                "OpenRouter keys start with 'sk-or-v1-...'. "
                "Please get your free key at https://openrouter.ai/keys and paste it in the key input box."
            )

        headers = {
            "Authorization": f"Bearer {clean_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://reliabletsr.enterprise.internal",
            "X-Title": "ReliableTSR-Audit",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1500,
        }

        resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            error_msg = resp.text
            try:
                error_msg = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                pass
            if resp.status_code == 401:
                raise RuntimeError(
                    f"Authentication Failed (401): {error_msg}. "
                    "Make sure you are using an OpenRouter key starting with 'sk-or-v1-...' generated at https://openrouter.ai/keys (not a HuggingFace, OpenAI, or account password)."
                )
            raise RuntimeError(f"OpenRouter API error ({resp.status_code}): {error_msg}")

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"Model returned no choices: {data}")

        choice = choices[0]
        msg = choice.get("message", {})
        raw_content = (
            msg.get("content")
            or msg.get("reasoning")
            or msg.get("reasoning_content")
            or choice.get("text")
            or ""
        )

        if isinstance(raw_content, list):
            text_blocks = [item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content]
            return "\n".join(text_blocks).strip()

        return str(raw_content).strip() if raw_content else ""

    def _query_ollama(self, b64_image: str, prompt: str) -> str:
        """Invoke local Ollama vision endpoint."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64_image],
                }
            ],
            "stream": False,
        }
        resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error ({resp.status_code}): {resp.text}")
        return resp.json().get("message", {}).get("content", "")
