"""ScrapeOps proxy fallback for pages our own fetchers cannot get.

Two roles:

1. **Fallback.** When httpx and the headless browser both come back empty, short, or
   holding an anti-bot interstitial, the request is replayed through the ScrapeOps
   Proxy API Aggregator, which rotates residential IPs and solves the usual
   Cloudflare/DataDome challenges. It costs credits, so it only ever runs after a
   real failure.

2. **First choice for sites that always block us.** Reddit is the current example:
   it rejects datacenter IPs outright, so there is no point paying the latency of a
   local attempt first. Reddit is also fetched through its public JSON endpoint
   rather than the HTML page — that returns the post body and the full comment tree
   as structured data, which is both cheaper and far more faithful than scraping the
   rendered feed.
"""

from __future__ import annotations

import asyncio
import json
import weakref
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from config import Config

SCRAPEOPS_ENDPOINT = "https://proxy.scrapeops.io/v1/"

# ScrapeOps retries internally for up to 2 minutes before giving up, so a shorter
# client timeout would abandon requests we have already been charged for.
SCRAPEOPS_TIMEOUT_S = 150

# ScrapeOps' generic anti-bot engine. Level 3 clears DataDome (G2, Capterra) where
# residential IPs alone still get a 500; level 4 costs more without helping on the
# sites this pipeline targets.
ANTIBOT_BYPASS_LEVEL = "generic_level_3"

# Comment trees can be thousands of nodes deep-linked; these keep one page bounded.
REDDIT_COMMENT_LIMIT = 200
REDDIT_MAX_DEPTH = 4

# Short pages whose opening text matches one of these are challenge screens, not
# content, so they are treated as failures worth retrying through the proxy.
BLOCK_MARKERS = (
    "just a moment",
    "attention required",
    "checking your browser",
    "enable javascript and cookies",
    "access denied",
    "403 forbidden",
    "are you a robot",
    "verify you are human",
    "captcha",
    "unusual traffic",
    "rate limit",
    "blocked",
)


class ScrapeOpsError(RuntimeError):
    pass


# ------------------------------------------------------------------ shared client

# Keyed by event loop, because every batch runs on a fresh one (see scraper.run_async)
# and a pooled connection belonging to a closed loop fails with "Event loop is closed".
_clients: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


async def _get_client() -> "httpx.AsyncClient":
    import httpx

    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            timeout=SCRAPEOPS_TIMEOUT_S,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        _clients[loop] = client
    return client


# --------------------------------------------------------------------- detection


def is_reddit(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == "reddit.com" or host.endswith(".reddit.com")


def needs_proxy_first(url: str) -> bool:
    """Sites that block every direct request, so a local attempt is wasted time."""
    return is_reddit(url)


def looks_blocked(text: str, cfg: Config) -> bool:
    t = (text or "").strip()
    if len(t) < cfg.thin_content_chars:
        return True
    head = t[:2000].lower()
    return any(m in head for m in BLOCK_MARKERS)


def available(cfg: Config) -> bool:
    return bool(cfg.scrapeops_enabled and cfg.scrapeops_api_key)


# ------------------------------------------------------------------ raw proxy get


async def _proxy_get(
    url: str,
    cfg: Config,
    *,
    render_js: bool,
    residential: bool = False,
    bypass: str = "",
) -> str:
    client = await _get_client()
    params: dict[str, Any] = {
        "api_key": cfg.scrapeops_api_key,
        "url": url,
        "country": cfg.scrapeops_country or None,
    }
    if render_js:
        params["render_js"] = "true"
        if cfg.scrapeops_wait_ms > 0:
            params["wait_time"] = cfg.scrapeops_wait_ms
    if residential:
        params["residential"] = "true"
    if bypass:
        params["bypass"] = bypass
    params = {k: v for k, v in params.items() if v is not None}

    resp = await client.get(SCRAPEOPS_ENDPOINT, params=params)
    if resp.status_code != 200:
        raise ScrapeOpsError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.text


async def _proxy_get_escalating(url: str, cfg: Config, *, render_js: bool) -> str:
    """Climb the price ladder only as far as the site forces us.

    ScrapeOps answers 500 when the current tier cannot get through. Each rung costs
    materially more than the last -- residential IPs more than datacenter, and the
    anti-bot bypass engine more again -- so each is only paid for after the cheaper
    one has actually failed.
    """
    attempts: list[tuple[str, dict]] = [("standard", {})]
    if cfg.scrapeops_residential_retry:
        attempts.append(("residential", {"residential": True}))
        attempts.append(("bypass", {"bypass": ANTIBOT_BYPASS_LEVEL}))

    errors: list[str] = []
    for name, kwargs in attempts:
        try:
            return await _proxy_get(url, cfg, render_js=render_js, **kwargs)
        except ScrapeOpsError as e:
            errors.append(f"{name}: {e}")
    raise ScrapeOpsError(" | ".join(errors))


# ----------------------------------------------------------------- reddit parsing


def reddit_json_url(url: str) -> str:
    """Reddit serves any listing, post, or profile as JSON by appending `.json`."""
    parts = urlparse(url)
    path = parts.path.rstrip("/")
    if not path.endswith(".json"):
        path += ".json"
    # Whatever the user pointed at is preserved -- dropping `?t=year` from a top
    # listing would quietly return a different set of posts.
    params = dict(parse_qsl(parts.query))
    # raw_json=1 stops HTML-escaping of quotes and ampersands in bodies.
    params.setdefault("raw_json", "1")
    params.setdefault("limit", str(REDDIT_COMMENT_LIMIT))
    params.setdefault("sort", "top")
    return urlunparse(parts._replace(path=path, query=urlencode(params), fragment=""))


def _coerce_json(text: str) -> Any:
    """Parse the proxy response, tolerating a browser-rendered JSON view."""
    body = text.strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    if body.startswith("<"):
        from bs4 import BeautifulSoup

        body = BeautifulSoup(body, "lxml").get_text("\n").strip()
        return json.loads(body)
    raise ScrapeOpsError("Reddit returned a non-JSON body")


def _fmt_post(d: dict) -> tuple[str, list[str]]:
    title = str(d.get("title") or "").strip()
    lines = [f"# {title}" if title else "# (untitled post)"]
    meta = [f"score {d.get('score', 0)}", f"{d.get('num_comments', 0)} comments"]
    if d.get("subreddit"):
        meta.insert(0, f"r/{d['subreddit']}")
    if d.get("author"):
        meta.insert(-2, f"u/{d['author']}")
    lines.append(" · ".join(meta))
    body = str(d.get("selftext") or "").strip()
    if body:
        lines += ["", body]
    elif d.get("url_overridden_by_dest"):
        lines += ["", f"Link: {d['url_overridden_by_dest']}"]
    return title, lines


def _fmt_comments(children: list[dict], depth: int = 0) -> list[str]:
    out: list[str] = []
    for child in children:
        if child.get("kind") != "t1":
            continue  # "more" stubs hold no text
        d = child.get("data") or {}
        body = str(d.get("body") or "").strip()
        if not body or body in ("[deleted]", "[removed]"):
            continue
        pad = "  " * depth
        out.append(f"{pad}- u/{d.get('author', '?')} ({d.get('score', 0)}): {body}")
        replies = d.get("replies")
        if depth < REDDIT_MAX_DEPTH and isinstance(replies, dict):
            kids = (replies.get("data") or {}).get("children") or []
            out += _fmt_comments(kids, depth + 1)
    return out


def format_reddit(payload: Any) -> tuple[str, str]:
    """(title, text) from Reddit's JSON, covering both post pages and listings."""
    # A post page is [post listing, comment listing]; anything else is a plain listing.
    if isinstance(payload, list) and payload:
        post_children = ((payload[0] or {}).get("data") or {}).get("children") or []
        if not post_children:
            raise ScrapeOpsError("Reddit JSON held no post")
        title, lines = _fmt_post(post_children[0].get("data") or {})
        comments: list[str] = []
        if len(payload) > 1:
            kids = ((payload[1] or {}).get("data") or {}).get("children") or []
            comments = _fmt_comments(kids)
        if comments:
            lines += ["", f"## Comments ({len(comments)})", *comments]
        return title, "\n".join(lines).strip()

    children = ((payload or {}).get("data") or {}).get("children") or []
    if not children:
        raise ScrapeOpsError("Reddit JSON held no posts")
    lines: list[str] = []
    for child in children:
        d = child.get("data") or {}
        _, block = _fmt_post(d)
        lines += block + [""]
    return "Reddit listing", "\n".join(lines).strip()


# ------------------------------------------------------------------- public fetch


async def fetch(url: str, cfg: Config) -> tuple[str, str, str]:
    """(title, text, fetch_method). Raises ScrapeOpsError on failure."""
    if not available(cfg):
        raise ScrapeOpsError("ScrapeOps is not configured")

    if is_reddit(url):
        # Reddit refuses the datacenter pool outright, so go straight to residential
        # rather than paying for an attempt that is known to fail.
        raw = await _proxy_get(
            reddit_json_url(url), cfg, render_js=False, residential=True
        )
        title, text = format_reddit(_coerce_json(raw))
        return title, text, "scrapeops-reddit"

    html = await _proxy_get_escalating(url, cfg, render_js=cfg.scrapeops_render_js)
    from pipeline.scraper import _extract_text

    title, text = await asyncio.to_thread(_extract_text, html, url)
    if not text:
        raise ScrapeOpsError("ScrapeOps returned no extractable text")
    return title, text, "scrapeops"
