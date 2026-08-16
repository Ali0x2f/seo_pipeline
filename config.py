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
DATA_DIR = ROOT / "data"

for _d in (SCHEMA_DIR, CACHE_DIR, RUNS_DIR, OUTPUT_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Where the storage backend choice is remembered between restarts. Kept out of .env so
# the UI can change it without rewriting a file the user hand-edits.
STORAGE_SETTINGS_FILE = ROOT / "storage.json"

# Edited system prompts, per scenario. Same reasoning as above: the UI owns this file.
PROMPTS_FILE = ROOT / "prompts.json"

DEFAULT_SQLITE_PATH = DATA_DIR / "runs.db"

# files | db | both. "both" keeps writing JSON while a database is being trialled.
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "files")
# Any SQLAlchemy URL: sqlite:///…, postgresql+psycopg://user:pw@host/db, mysql+pymysql://…
DATABASE_URL = os.getenv("DATABASE_URL", "")


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
    provider: str = os.getenv("LLM_PROVIDER", "deepseek")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
    )
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model: str = os.getenv("LLM_MODEL", "deepseek-v4-flash")
    temperature: float = _env_float("LLM_TEMPERATURE", 0.1)
    max_output_tokens: int = _env_int("LLM_MAX_OUTPUT_TOKENS", 8000)
    llm_concurrency: int = _env_int("LLM_CONCURRENCY", 4)

    # --- Scraping ---
    scrape_concurrency: int = _env_int("SCRAPE_CONCURRENCY", 4)
    page_timeout_ms: int = _env_int("PAGE_TIMEOUT_MS", 60000)
    settle_delay_s: float = _env_float("SETTLE_DELAY_S", 2.5)
    max_scrape_chars: int = _env_int("MAX_SCRAPE_CHARS", 50000)
    thin_content_chars: int = _env_int("THIN_CONTENT_CHARS", 600)
    request_timeout_s: int = _env_int("REQUEST_TIMEOUT_S", 30)
    user_agent: str = os.getenv(
        "USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    )

    # --- ScrapeOps proxy fallback ---
    # Used only after our own fetchers fail or get challenged, plus up front for sites
    # that reject datacenter IPs outright (Reddit). Billed per request, so it stays off
    # the happy path.
    scrapeops_api_key: str = os.getenv("SCRAPEOPS_API_KEY", "")
    scrapeops_enabled: bool = os.getenv("SCRAPEOPS_ENABLED", "1") not in (
        "0",
        "false",
        "False",
    )
    scrapeops_render_js: bool = os.getenv("SCRAPEOPS_RENDER_JS", "1") not in (
        "0",
        "false",
        "False",
    )
    scrapeops_country: str = os.getenv("SCRAPEOPS_COUNTRY", "us")
    scrapeops_wait_ms: int = _env_int("SCRAPEOPS_WAIT_MS", 3000)
    # Allows escalating to residential IPs and then the anti-bot bypass engine when the
    # standard pool is refused. Each step costs more credits, so each is only paid for
    # after the cheaper one has failed.
    scrapeops_residential_retry: bool = os.getenv(
        "SCRAPEOPS_RESIDENTIAL_RETRY", "1"
    ) not in ("0", "false", "False")

    # --- Authentication ---
    # Gates the whole UI behind a login form. Empty means no credentials configured, so
    # the app keeps its previous open-by-default behaviour -- set one to require sign-in.
    auth_username: str = os.getenv("AUTH_USERNAME", "admin")
    auth_password: str = os.getenv("AUTH_PASSWORD", "")
    # "alice:secret1,bob:secret2" for more than one account; each pair wins over the
    # single username/password above for the user it names.
    auth_users: str = os.getenv("AUTH_USERS", "")

    # --- Behaviour ---
    use_cache: bool = os.getenv("USE_CACHE", "1") not in ("0", "false", "False")
    reconcile_batch_size: int = _env_int("RECONCILE_BATCH_SIZE", 6)

    # --- Prompts ---
    # "general" writes about the article's subject product; "tools" profiles each
    # alternative it is compared against. A schema declares which one it belongs to;
    # this is the fallback for schemas that do not.
    scenario: str = os.getenv("PROMPT_SCENARIO", "general")
    # Unsaved edits from the Advanced tab. Empty means "use the saved/default prompt".
    extract_prompt_override: str = ""
    merge_prompt_override: str = ""
    web_prompt_override: str = ""

    # --- Web conflict resolution ---
    # Off by default: it costs a per-search fee on top of tokens, and only OpenAI and
    # Anthropic expose a hosted search tool.
    web_check_conflicts: bool = os.getenv("WEB_CHECK_CONFLICTS", "0") not in (
        "0",
        "false",
        "False",
        "",
    )
    web_check_max_searches: int = _env_int("WEB_CHECK_MAX_SEARCHES", 4)

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
