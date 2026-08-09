"""Stage 2: turn each scraped page into per-field claims about one product.

Two decisions drive the design.

Subject isolation. Reference pages for "alternatives" content are overwhelmingly
"A vs B" comparisons, so the dominant accuracy risk is attributing the competitor's
traits to the product under study. The prompt therefore names the subject repeatedly
and forbids describing anything else.

Provenance. Extraction happens per source page rather than per product, so every claim
keeps the URL it came from. Each claim must also carry a verbatim supporting quote,
which we then check against the page text. An unverifiable quote is the best cheap
signal that a claim was invented.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from config import Config
from pipeline.cache import DiskCache, make_key
from pipeline.models import FieldClaim, ScrapedPage, SourceExtraction
from pipeline.prompts import resolve as resolve_prompt
from pipeline.providers import BaseProvider, LLMError, Usage
from pipeline.schema import FieldSpec, SchemaSpec

EXTRACT_CACHE_VERSION = 4

ProgressCb = Callable[[int, int, str], None] | None


def system_prompt(spec: SchemaSpec, cfg: Config) -> str:
    return resolve_prompt("extract", spec.scenario, cfg.extract_prompt_override)


def _norm_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def build_extraction_schema(fields: list[FieldSpec]) -> dict:
    """JSON Schema compatible with OpenAI strict mode.

    Strict mode requires every property to be listed in `required` and forbids
    validation keywords such as maxItems, so list length is steered by the prompt and
    enforced afterwards.
    """
    props: dict[str, dict] = {}
    for f in fields:
        props[f.key] = {
            "type": "object",
            "description": f.label,
            "properties": {
                "found": {
                    "type": "boolean",
                    "description": "True only if the page text addresses this field "
                    "for the subject product.",
                },
                "value": f.json_value_property(),
                "quote": {
                    "type": "string",
                    "description": "A short verbatim excerpt from the page text "
                    "supporting the value. Empty when found is false.",
                },
            },
            "required": ["found", "value", "quote"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "properties": props,
                "required": list(props.keys()),
                "additionalProperties": False,
            }
        },
        "required": ["fields"],
        "additionalProperties": False,
    }


def build_user_prompt(
    spec: SchemaSpec, fields: list[FieldSpec], page: ScrapedPage, product: str
) -> str:
    lines = [
        f"SUBJECT {spec.entity_label.upper()}: {product}",
        "",
        f"Extract the fields below about {product} only.",
        "",
        "FIELDS",
    ]
    for f in fields:
        shape = {
            "list": f"list of up to {f.max_items} short points",
            "short_text": "one short phrase",
            "prose": "2-4 sentences",
        }[f.shape.value]
        section = f"{f.section} > " if f.section.strip() else ""
        lines.append(f"- {f.key} ({section}{f.label}) [{shape}]: {f.prompt_line()}")

    # Analyst-supplied evidence is quoted material the brief attached to a field. It is
    # evidence like any page text, so a claim may cite it -- but it is not the page, so
    # it is fenced separately and the quote check knows about it.
    extras = [(f, f.custom_input.strip()) for f in fields if f.custom_input.strip()]
    if extras:
        lines += ["", "ANALYST-SUPPLIED EVIDENCE (counts as permitted evidence):"]
        for f, text in extras:
            lines += [
                f"-------- BEGIN {f.key} NOTES --------",
                text,
                f"-------- END {f.key} NOTES --------",
            ]

    src = f"{page.title} <{page.url}>" if page.title else page.url
    lines += [
        "",
        f"PAGE SOURCE: {src}",
        "PAGE TEXT (the only permitted evidence):",
        "-------- BEGIN PAGE TEXT --------",
        page.text,
        "-------- END PAGE TEXT --------",
        "",
        f"Reminder: describe {product} and nothing else. Mark found=false wherever the "
        "page is silent.",
    ]
    return "\n".join(lines)


def _coerce_values(raw, spec_field: FieldSpec) -> list[str]:
    """Normalise the model's value into a list of clean strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = [str(x) for x in raw]
    else:
        items = [str(raw)]

    out: list[str] = []
    for it in items:
        t = it.strip().strip("•").strip()
        if not t or t.lower() in {"n/a", "na", "none", "unknown", "not specified", "-"}:
            continue
        if t not in out:
            out.append(t)
    if spec_field.is_list:
        return out[: spec_field.max_items]
    return out[:1] if out else []


def parse_extraction(
    data: dict, fields: list[FieldSpec], page_text: str
) -> dict[str, FieldClaim]:
    payload = (data or {}).get("fields") or {}
    # Analyst notes were shown to the model as evidence, so a quote drawn from them is
    # genuine support and must not be flagged as invented.
    notes = " ".join(f.custom_input for f in fields if f.custom_input.strip())
    haystack = _norm_for_match(f"{page_text}\n{notes}" if notes else page_text)
    claims: dict[str, FieldClaim] = {}

    for f in fields:
        node = payload.get(f.key) or {}
        found = bool(node.get("found"))
        values = _coerce_values(node.get("value"), f)
        quote = str(node.get("quote") or "").strip()

        if not values:
            found = False

        verified = False
        if quote:
            nq = _norm_for_match(quote)
            # Long quotes get a prefix check: models often truncate with an ellipsis.
            verified = nq in haystack or (len(nq) > 60 and nq[:60] in haystack)

        claims[f.key] = FieldClaim(
            field_key=f.key,
            found=found,
            values=values if found else [],
            quote=quote,
            quote_verified=verified,
        )
    return claims


def extract_page(
    page: ScrapedPage,
    spec: SchemaSpec,
    provider: BaseProvider,
    cfg: Config,
    cache: DiskCache,
) -> SourceExtraction:
    canonical, _ = spec.resolve_node(page.node)
    fields = spec.fields_for_node(canonical)

    base = SourceExtraction(
        product=page.product, url=page.url, node=page.node, model=provider.model
    )
    if not page.success or not page.text.strip():
        base.error = page.error or "No page content"
        return base
    if not fields:
        base.error = f"Node {page.node!r} feeds no fields in this schema"
        return base

    system = system_prompt(spec, cfg)
    fingerprint = make_key(
        *[f"{f.key}|{f.shape}|{f.prompt_line()}|{f.custom_input}" for f in fields]
    )
    ck = make_key(
        "extract",
        EXTRACT_CACHE_VERSION,
        provider.name,
        provider.model,
        cfg.temperature,
        system,
        page.product,
        page.url,
        fingerprint,
        page.text,
    )
    hit = cache.get(ck)
    if hit is not None:
        base.claims = parse_extraction(hit, fields, page.text)
        base.from_cache = True
        return base

    schema = build_extraction_schema(fields)
    user = build_user_prompt(spec, fields, page, page.product)
    try:
        resp = provider.complete_json(
            system,
            user,
            schema,
            max_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
        )
    except LLMError as e:
        base.error = str(e)
        return base
    except Exception as e:  # noqa: BLE001
        base.error = f"{type(e).__name__}: {e}"
        return base

    cache.set(ck, resp.data)
    base.claims = parse_extraction(resp.data, fields, page.text)
    base.prompt_tokens = resp.prompt_tokens
    base.completion_tokens = resp.completion_tokens
    return base


def extract_pages(
    pages: list[ScrapedPage],
    spec: SchemaSpec,
    provider: BaseProvider,
    cfg: Config,
    progress: ProgressCb = None,
) -> tuple[list[SourceExtraction], Usage]:
    cache = DiskCache("extract", enabled=cfg.use_cache)
    usage = Usage()
    results: list[SourceExtraction] = [None] * len(pages)  # type: ignore[list-item]

    def work(i: int) -> tuple[int, SourceExtraction]:
        return i, extract_page(pages[i], spec, provider, cfg, cache)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.llm_concurrency)) as pool:
        futures = [pool.submit(work, i) for i in range(len(pages))]
        for fut in as_completed(futures):
            i, ext = fut.result()
            results[i] = ext
            done += 1
            if ext.error:
                usage.errors += 1
            elif ext.from_cache:
                usage.cached += 1
            else:
                usage.calls += 1
                usage.prompt_tokens += ext.prompt_tokens
                usage.completion_tokens += ext.completion_tokens
            if progress:
                progress(done, len(pages), f"{ext.product} · {ext.url[:60]}")

    return results, usage
