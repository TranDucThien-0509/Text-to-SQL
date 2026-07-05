"""
OpenRouterClient – thread-safe LLM client for OpenRouter API.

Features:
  - Exponential back-off retries
  - SQL post-processing (strip fences, fix double SELECT, etc.)
  - Token usage tracking
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional, Tuple

import requests

from text2sql.core.config import PipelineConfig

logger = logging.getLogger(__name__)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    """
    Calls OpenRouter chat completions and post-processes the raw output
    into a clean, executable SQL string.
    """

    def __init__(self, config: PipelineConfig) -> None:
        if not config.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Export it or pass via T2SQL_OPENROUTER_API_KEY env var."
            )
        self._config = config
        self._headers = {
            "Authorization": f"Bearer {config.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/text2sql-research",
        }
        # Cumulative token counters (thread-safe via GIL for int ops)
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """
        Call the LLM and return a post-processed SQL string.
        Falls back to 'SELECT 1' after all retries are exhausted.
        """
        sql, _ = self.generate_with_usage(prompt)
        return sql

    def generate_with_usage(self, prompt: str) -> Tuple[str, dict]:
        """
        Return (sql_string, usage_dict) where usage_dict has
        keys 'prompt_tokens' and 'completion_tokens'.
        """
        payload = {
            "model": self._config.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._config.llm_temperature,
            "max_tokens": 512,
        }

        delay = self._config.api_retry_delay
        for attempt in range(1, self._config.api_retries + 1):
            try:
                resp = requests.post(
                    _API_URL,
                    headers=self._headers,
                    json=payload,
                    timeout=self._config.api_timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                raw = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                pt = usage.get("prompt_tokens", 0)
                ct = usage.get("completion_tokens", 0)
                self.total_prompt_tokens += pt
                self.total_completion_tokens += ct

                sql = self._post_process(raw)
                return sql, {"prompt_tokens": pt, "completion_tokens": ct}

            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response else "?"
                logger.warning(
                    "[LLM] HTTP %s on attempt %d/%d", status, attempt, self._config.api_retries
                )
                if status == 429:   # rate-limited – longer back-off
                    delay *= 2
            except Exception as exc:
                logger.warning(
                    "[LLM] Error on attempt %d/%d: %s", attempt, self._config.api_retries, exc
                )

            if attempt < self._config.api_retries:
                time.sleep(delay)
                delay = min(delay * 1.5, 30.0)

        logger.error("[LLM] All %d retries exhausted. Returning fallback SQL.", self._config.api_retries)
        return "SELECT 1", {"prompt_tokens": 0, "completion_tokens": 0}

    # ── Post-processing ────────────────────────────────────────────────────

    @staticmethod
    def _post_process(raw: str) -> str:
        text = raw.strip()

        # Strip markdown code fences
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else parts[0]
            if text.lower().startswith("sql"):
                text = text[3:]
            text = text.strip()

        # Keep only first statement
        if ";" in text:
            text = text.split(";")[0]

        # Remove stray backticks and double-quotes used as identifiers
        text = text.replace("`", "").replace('"', "")

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Fix "SELECT SELECT ..." double prefix (PromptBuilder ends with "SELECT ")
        while re.match(r"^SELECT\s+SELECT\b", text, re.IGNORECASE):
            text = text[7:].strip()

        # Ensure starts with SELECT
        if not text.upper().startswith("SELECT"):
            text = "SELECT " + text

        return text