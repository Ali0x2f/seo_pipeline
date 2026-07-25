"""Page fetching via crawl4ai (headless Chromium) with a static pre-check.

Strategy:
  1. Run a fast static fetch (httpx + trafilatura) for every pending URL at high
     concurrency.  Many pages serve usable content without JS — these skip the
     expensive browser step entirely.
  2. Only launch headless Chromium for URLs where static extraction returned
     less than the configured character budget.  This is the minority (JS-only
     pricing pages, SPAs, etc.).
  3. Per-domain serialisation uses a lock but the politeness delay fires
     *after* releasing it, so other domains are never blocked.

Notes:
* JS rendering is not optional for pricing tables, but we only pay for it when
  we actually need it.
* One browser instance is reused for the whole batch. Launching Chromium per URL
  costs ~1.5s each and exhausts handles quickly.
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
STATIC_CONCURRENCY_MULTIPLIER = 3  # static reqs are cheap — run more in parallel

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
        except BaseException as e:  # surfaced to the caller below
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


# ------------------------------------------------------------- static (pre-check + fallback)

# Shared httpx client for the static pre-checks — one pool for the whole batch avoids
# tearing down and re-establishing connections on every URL.
_static_client: "httpx.AsyncClient | None" = None


async def _get_static_client(cfg: Config) -> "httpx.AsyncClient":
    global _static_client
    if _static_client is None or _static_client.is_closed:
        import httpx

        _static_client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=cfg.request_timeout_s,
            headers={"User-Agent": cfg.user_agent, "Accept-Language": "en-US,en;q=0.9"},
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30,
            ),
        )
    return _static_client


def _extract_text(html: str, url: str) -> tuple[str, str]:
    """(title, text) via trafilatura with a BeautifulSoup fallback for thin pages."""
    import trafilatura

    text = (
        trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=False,
        )
        or ""
    )

    title = ""
    meta = trafilatura.extract_metadata(html)
    if meta and getattr(meta, "title", None):
        title = meta.title or ""

    if len(text) < 600:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(
            ["script", "style", "noscript", "nav", "footer", "header", "aside"]
        ):
            tag.decompose()
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()
        lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
        bs_text = "\n".join(lines)
        if len(bs_text) > len(text):
            text = bs_text

    return title, text.strip()


async def _fetch_static_async(url: str, cfg: Config) -> tuple[str, str]:
    """(title, text) via async httpx + trafilatura, run in a thread to keep the event
    loop free for the browser."""
    client = await _get_static_client(cfg)
    resp = await client.get(url)
    resp.raise_for_status()
    return await asyncio.to_thread(_extract_text, resp.text, url)


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
    """Fetch every distinct URL once.  Returns url -> payload dict.

    Two-phase approach:
      1. Static pre-check — run at high concurrency for ALL pending URLs.
         URLs that return >= max_scrape_chars of usable text skip the browser.
      2. Browser phase — only the remaining JS-heavy URLs go through Chromium.
    """
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    out: dict[str, dict] = {}
    pending: list[str] = []

    # ── cache check ────────────────────────────────────────────────
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

    # ── phase 1: static pre-check (cheap, high concurrency) ────────
    # Many pages (docs, blog posts, product pages) serve their content
    # server-side.  We can grab it with a plain HTTP call and skip the
    # expensive browser launch entirely for those URLs.
    #
    # While the static checks run, warm up the browser in the background
    # so it's ready immediately if any URLs need it.  This eliminates the
    # ~1-2s cold-start penalty from the critical path.
    static_sem = asyncio.Semaphore(
        max(1, cfg.scrape_concurrency * STATIC_CONCURRENCY_MULTIPLIER)
    )
    lock = asyncio.Lock()

    # Pre-configure the browser so we can launch it concurrently.
    browser_cfg = BrowserConfig(
        headless=True,
        verbose=False,
        user_agent=cfg.user_agent,
        extra_args=["--disable-blink-features=AutomationControlled", "--disable-gpu"],
    )

    # Start the browser in the background — runs in parallel with static checks.
    browser_task = asyncio.create_task(AsyncWebCrawler(config=browser_cfg).__aenter__())

    async def _static_one(url: str) -> dict | None:
        """Returns a payload if static was sufficient, None if we need the browser."""
        payload: dict = {
            "url": url,
            "success": False,
            "title": "",
            "text": "",
            "fetch_method": "",
            "error": None,
        }
        try:
            async with static_sem:
                title, text = await _fetch_static_async(url, cfg)
            if text:
                payload["title"] = title
                payload["text"] = text
                payload["success"] = True
                payload["fetch_method"] = "httpx-trafilatura"
        except Exception as e:
            payload["error"] = f"static: {type(e).__name__}: {e}"

        # If static gave us enough content, accept it and skip the browser.
        if payload["success"] and len(payload["text"]) >= cfg.max_scrape_chars:
            cache.set(
                make_key("scrape", SCRAPE_CACHE_VERSION, url, cfg.max_scrape_chars),
                payload,
            )
            async with lock:
                out[url] = payload
                nonlocal done
                done += 1
                if progress:
                    progress(done, total, f"static ✓ {url}")
            return payload
        return None

    static_tasks = [asyncio.create_task(_static_one(u)) for u in pending]
    static_results = await asyncio.gather(*static_tasks)

    # Collect URLs that still need the browser.
    browser_pending: list[str] = []
    static_payloads: dict[str, dict] = {}
    for url, result in zip(pending, static_results):
        if result is not None:
            # static was sufficient — already stored in `out`
            static_payloads[url] = result
        else:
            browser_pending.append(url)

    if not browser_pending:
        # Browser was warming up but nobody needs it — shut it down.
        crawler = await browser_task
        await crawler.__aexit__(None, None, None)
        return out

    if progress:
        progress(done, total, f"static done — {len(browser_pending)} need browser")

    # ── phase 2: browser for JS-heavy pages ────────────────────────
    # The browser has been warming up since phase 1 started — it should
    # be ready now with zero additional cold-start cost.
    crawler = await browser_task
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=cfg.page_timeout_ms,
        wait_until="networkidle",
        delay_before_return_html=cfg.settle_delay_s,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.45, threshold_type="dynamic"
            )
        ),
        excluded_tags=["script", "style", "noscript", "form"],
        remove_overlay_elements=True,
        scan_full_page=True,
        word_count_threshold=5,
    )

    sem = asyncio.Semaphore(max(1, cfg.scrape_concurrency))
    domain_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    try:

        async def _browser_one(url: str) -> None:
            nonlocal done
            payload: dict = {
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
                    # Delay AFTER releasing the lock so other domains proceed.
                    await asyncio.sleep(PER_DOMAIN_DELAY_S)

                if getattr(res, "success", False):
                    text = _pick_text(res, cfg)
                    payload["title"] = (getattr(res, "metadata", None) or {}).get(
                        "title", ""
                    ) or ""
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

            # Merge with static result if we have one — prefer browser content
            # but keep static as fallback when the browser returned nothing useful.
            static = static_payloads.get(url)
            if static and len(payload["text"]) < len(static["text"]):
                payload["title"] = payload["title"] or static["title"]
                payload["text"] = static["text"]
                payload["fetch_method"] = (
                    "crawl4ai+static" if payload["success"] else "httpx-trafilatura"
                )
                payload["success"] = bool(payload["text"])
                if payload["success"]:
                    payload["error"] = None

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

        await asyncio.gather(*(_browser_one(u) for u in browser_pending))

    finally:
        await crawler.__aexit__(None, None, None)

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

    fetched = (
        run_async(_scrape_batch(distinct, cfg, cache, progress)) if distinct else {}
    )

    pages: list[ScrapedPage] = []
    for r in rows:
        cu = canon.get(r.url, "")
        payload = fetched.get(cu)
        if payload is None:
            pages.append(
                ScrapedPage(
                    url=r.url,
                    node=r.node,
                    product=r.product,
                    success=False,
                    error="Invalid or empty URL",
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
