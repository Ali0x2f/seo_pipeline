"""Central configuration. Environment supplies defaults; the UI can override per run."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
SCHEMA_DIR = ROOT / "schemas"
CACHE_DIR = ROOT / ".cache"
RUNS_DIR = ROOT / "runs"
OUTPUT_DIR = ROOT / "output"

for _d in (SCHEMA_DIR, CACHE_DIR, RUNS_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "") or default)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "") or default)
    except ValueError:
        return default


@dataclass
class Config:
    # --- LLM ---
    provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature: float = _env_float("LLM_TEMPERATURE", 0.1)
    max_output_tokens: int = _env_int("LLM_MAX_OUTPUT_TOKENS", 8000)
    llm_concurrency: int = _env_int("LLM_CONCURRENCY", 4)

    # --- Scraping ---
    scrape_concurrency: int = _env_int("SCRAPE_CONCURRENCY", 4)
    page_timeout_ms: int = _env_int("PAGE_TIMEOUT_MS", 60000)
    settle_delay_s: float = _env_float("SETTLE_DELAY_S", 2.5)
    max_scrape_chars: int = _env_int("MAX_SCRAPE_CHARS", 40000)
    thin_content_chars: int = _env_int("THIN_CONTENT_CHARS", 600)
    request_timeout_s: int = _env_int("REQUEST_TIMEOUT_S", 30)
    user_agent: str = os.getenv(
        "USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    )

    # --- Behaviour ---
    use_cache: bool = os.getenv("USE_CACHE", "1") not in ("0", "false", "False")
    reconcile_batch_size: int = _env_int("RECONCILE_BATCH_SIZE", 6)

    def key_for(self, provider: str | None = None) -> str:
        p = (provider or self.provider).lower()
        if p == "anthropic":
            return self.anthropic_api_key
        if p == "deepseek":
            return self.deepseek_api_key
        if p == "ollama":
            return "ollama"
        return self.openai_api_key

    def base_url_for(self, provider: str | None = None) -> str:
        p = (provider or self.provider).lower()
        if p == "deepseek":
            return self.deepseek_base_url
        if p == "ollama":
            return self.ollama_base_url
        return self.openai_base_url


config = Config()
