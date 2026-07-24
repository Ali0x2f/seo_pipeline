"""Page fetching via crawl4ai (headless Chromium) with a static fallback.

Notes that matter:
* JS rendering is not optional. Pricing pages in particular build their tables client
  side; a plain HTTP GET returns navigation chrome and no numbers.
* One browser instance is reused for the whole batch. Launching Chromium per URL costs
  ~1.5s each and exhausts handles quickly.
* Requests to the same host are serialised with a small delay. Politeness, and it avoids
  tripping rate limits mid-run.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections import defaultdict
from typing import Callable, Iterable
from urllib.parse import urlparse, urlunparse

from config import Config
from pipeline.cache import DiskCache, make_key
from pipeline.models import InputRow, ScrapedPage

SCRAPE_CACHE_VERSION = 4
PER_DOMAIN_DELAY_S = 1.0

ProgressCb = Callable[[int, int, str], None] | None


# --------------------------------------------------------------------------- utils

def normalize_url(url: str) -> str:
    """Canonical form used for fetching and cache keys. Fragments never change what
    the server returns, so they must not fragment the cache."""
    u = (url or "").strip()
    if not u:
        return ""
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    parts = urlparse(u)
    return urlunparse(parts._replace(fragment=""))


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except ValueError:
        return ""


def run_async(coro):
    """Run a coroutine from Streamlit's synchronous script thread.

    Playwright on Windows needs a Proactor event loop, and Streamlit may already own
    the current thread's loop, so we always use a fresh thread with a fresh loop.
    """
    result: dict = {}

    def _target() -> None:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["value"] = loop.run_until_complete(coro)
        except BaseException as e:            # surfaced to the caller below
            result["error"] = e
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


# ------------------------------------------------------------------ static fallback

def _fetch_static(url: str, cfg: Config) -> tuple[str, str]:
    """(title, text) via plain HTTP + trafilatura. Used when the browser yields little."""
    import httpx
    import trafilatura

    with httpx.Client(
        follow_redirects=True,
        timeout=cfg.request_timeout_s,
        headers={"User-Agent": cfg.user_agent, "Accept-Language": "en-US,en;q=0.9"},
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        html = resp.text

    text = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=True, favor_precision=False
    ) or ""

    title = ""
    meta = trafilatura.extract_metadata(html)
    if meta and getattr(meta, "title", None):
        title = meta.title or ""

    if len(text) < cfg.thin_content_chars:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
            tag.decompose()
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()
        lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
        bs_text = "\n".join(lines)
        if len(bs_text) > len(text):
            text = bs_text

    return title, text.strip()


# ------------------------------------------------------------------- browser fetch

def _pick_text(result, cfg: Config) -> str:
    """Choose between crawl4ai's full markdown and its boilerplate-pruned variant.

    Prefer the full text whenever it fits the budget. Pruning can silently discard real
    content -- observed dropping pricing tiers on some pages -- and that failure mode is
    far worse than carrying some navigation noise, which the LLM ignores for a negligible
    number of tokens. Pruning is therefore only used to avoid blind truncation.
    """
    md = getattr(result, "markdown", None)
    raw = str(getattr(md, "raw_markdown", md) or "").strip()
    fit = str(getattr(md, "fit_markdown", "") or "").strip()

    if len(raw) <= cfg.max_scrape_chars:
        return raw or fit
    if len(fit) >= cfg.thin_content_chars:
        return fit
    return raw


async def _scrape_batch(
    urls: list[str],
    cfg: Config,
    cache: DiskCache,
    progress: ProgressCb = None,
) -> dict[str, dict]:
    """Fetch every distinct URL once. Returns url -> payload dict."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    out: dict[str, dict] = {}
    pending: list[str] = []

    for u in urls:
        ck = make_key("scrape", SCRAPE_CACHE_VERSION, u, cfg.max_scrape_chars)
        hit = cache.get(ck)
        if hit is not None:
            hit["fetch_method"] = "cache"
            out[u] = hit
        else:
            pending.append(u)

    total = len(urls)
    done = len(out)
    if progress:
        progress(done, total, f"{done} from cache")

    if not pending:
        return out

    browser_cfg = BrowserConfig(
        headless=True,
        verbose=False,
        user_agent=cfg.user_agent,
        extra_args=["--disable-blink-features=AutomationControlled", "--disable-gpu"],
    )
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,          # we own caching
        page_timeout=cfg.page_timeout_ms,
        wait_until="networkidle",             # critical for client-rendered pricing
        delay_before_return_html=cfg.settle_delay_s,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.45, threshold_type="dynamic")
        ),
        excluded_tags=["script", "style", "noscript", "form"],
        remove_overlay_elements=True,         # dismisses cookie/consent walls
        scan_full_page=True,                  # triggers lazy-loaded sections
        word_count_threshold=5,
    )

    sem = asyncio.Semaphore(max(1, cfg.scrape_concurrency))
    domain_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    lock = asyncio.Lock()

    async with AsyncWebCrawler(config=browser_cfg) as crawler:

        async def one(url: str) -> None:
            nonlocal done
            payload = {
                "url": url,
                "success": False,
                "title": "",
                "text": "",
                "fetch_method": "",
                "error": None,
            }
            try:
                async with sem:
                    async with domain_locks[domain_of(url)]:
                        res = await crawler.arun(url=url, config=run_cfg)
                        await asyncio.sleep(PER_DOMAIN_DELAY_S)

                if getattr(res, "success", False):
                    text = _pick_text(res, cfg)
                    payload["title"] = (
                        (getattr(res, "metadata", None) or {}).get("title", "") or ""
                    )
                    payload["text"] = text
                    payload["fetch_method"] = "crawl4ai"
                    payload["success"] = bool(text)
                    if not text:
                        payload["error"] = "Browser returned no extractable text"
                else:
                    payload["error"] = (
                        getattr(res, "error_message", None) or "Browser fetch failed"
                    )
            except Exception as e:
                payload["error"] = f"{type(e).__name__}: {e}"

            # Fall back to static extraction when the browser path is thin or failed.
            if len(payload["text"]) < cfg.thin_content_chars:
                try:
                    title, text = await asyncio.to_thread(_fetch_static, url, cfg)
                    if len(text) > len(payload["text"]):
                        payload["title"] = payload["title"] or title
                        payload["text"] = text
                        payload["fetch_method"] = (
                            "httpx-trafilatura" if not payload["success"]
                            else "crawl4ai+static"
                        )
                        payload["success"] = bool(text)
                        payload["error"] = None if text else payload["error"]
                except Exception as e:
                    if not payload["success"]:
                        payload["error"] = f"{payload['error']} | static: {type(e).__name__}: {e}"

            if payload["success"]:
                cache.set(
                    make_key("scrape", SCRAPE_CACHE_VERSION, url, cfg.max_scrape_chars),
                    payload,
                )

            async with lock:
                out[url] = payload
                done += 1
                if progress:
                    progress(done, total, url)

        await asyncio.gather(*(one(u) for u in pending))

    return out


# ------------------------------------------------------------------------ public API

def scrape_rows(
    rows: Iterable[InputRow],
    cfg: Config,
    progress: ProgressCb = None,
) -> list[ScrapedPage]:
    """Scrape every input row, fetching each distinct URL only once."""
    rows = list(rows)
    cache = DiskCache("scrape", enabled=cfg.use_cache)

    canon: dict[str, str] = {}
    for r in rows:
        canon[r.url] = normalize_url(r.url)
    distinct = sorted({v for v in canon.values() if v})

    fetched = run_async(_scrape_batch(distinct, cfg, cache, progress)) if distinct else {}

    pages: list[ScrapedPage] = []
    for r in rows:
        cu = canon.get(r.url, "")
        payload = fetched.get(cu)
        if payload is None:
            pages.append(
                ScrapedPage(
                    url=r.url, node=r.node, product=r.product,
                    success=False, error="Invalid or empty URL",
                )
            )
            continue

        text = payload.get("text") or ""
        truncated = False
        if len(text) > cfg.max_scrape_chars:
            text = text[: cfg.max_scrape_chars]
            truncated = True

        pages.append(
            ScrapedPage(
                url=r.url,
                node=r.node,
                product=r.product,
                success=bool(payload.get("success")) and bool(text),
                title=payload.get("title") or "",
                text=text,
                char_count=len(text),
                fetch_method=payload.get("fetch_method") or "",
                truncated=truncated,
                error=payload.get("error"),
            )
        )
    return pages
