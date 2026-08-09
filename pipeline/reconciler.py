"""Stage 3: merge per-source claims into one value per product per field.

The naive approach -- joining every source's text into one cell -- produces three
problems this module exists to solve:

* Near-duplicates. Five reviews all say self-hosting is operationally heavy, in five
  different phrasings. The reader wants one point, not five.
* Contradictions. Two sources give two different starting prices, and blindly
  concatenating them hides the disagreement inside a plausible-looking cell. Conflicts
  are flagged for human review instead.
* Lost attribution. Merged prose with no record of which URL supported which claim
  cannot be fact-checked. Contributing sources are carried through.

Mechanical merging is available for a zero-cost pass; the LLM merge is the default
because only it can spot semantic duplicates and genuine contradictions.

Flagging a conflict tells the reviewer the sources disagree but not who is right, and
the scraped pages cannot settle it -- they are the disagreement. Optional web
arbitration takes each conflicting field to the vendor's hosted search tool, which
checks live pages (typically the vendor's own) and returns a verdict with citations.
It runs only on fields already marked as conflicts, so cost scales with disagreement
rather than schema size.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from config import Config
from pipeline.cache import DiskCache, make_key
from pipeline.models import (
    ProductResult,
    ReconciledField,
    SourceExtraction,
    WebVerdict,
)
from pipeline.prompts import resolve as resolve_prompt
from pipeline.providers import BaseProvider, Usage
from pipeline.schema import FieldSpec, SchemaSpec

RECONCILE_CACHE_VERSION = 3
WEB_CHECK_CACHE_VERSION = 2
ProgressCb = Callable[[int, int, str], None] | None


def merge_system_prompt(spec: SchemaSpec, cfg: Config) -> str:
    return resolve_prompt("merge", spec.scenario, cfg.merge_prompt_override)


def web_system_prompt(spec: SchemaSpec, cfg: Config) -> str:
    return resolve_prompt("web", spec.scenario, cfg.web_prompt_override)


def _norm(s: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def _digit_signature(s: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", s)


def _is_duplicate(a: str, b: str) -> bool:
    """Conservative duplicate test for already-normalised strings.

    Character-level similarity is deliberately avoided here. "supports 400 apps" and
    "supports 500 apps" are 0.94 similar by character but state different facts, so
    edit-distance measures silently destroy data. Two rules keep this safe:

    * Differing numbers always mean differing claims. Checked first, and it vetoes
      everything else.
    * Otherwise require exact equality, containment, or near-identical word sets, so
      distinguishing words like "self-hosted"/"cloud" prevent a merge.

    Semantic paraphrases ("hard to self-host" vs "self-hosting is operationally heavy")
    are intentionally NOT merged here; only the LLM merge can judge those safely.
    """
    if a == b:
        return True
    if _digit_signature(a) != _digit_signature(b):
        return False
    if a in b or b in a:
        return True
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.9


def _dedupe_points(values: list[str], limit: int) -> list[str]:
    """Drop exact repeats and containment duplicates, keeping the richest phrasing."""
    kept: list[str] = []
    norms: list[str] = []
    for v in values:
        c = (v or "").strip()
        if not c:
            continue
        n = _norm(c)
        if not n:
            continue
        for i, existing in enumerate(norms):
            if _is_duplicate(n, existing):
                if len(c) > len(kept[i]):  # keep whichever carries more detail
                    kept[i], norms[i] = c, n
                break
        else:
            kept.append(c)
            norms.append(n)
        if len(kept) >= limit:
            break
    return kept


# --------------------------------------------------------------------------- gather


def _collect(
    extractions: list[SourceExtraction], spec: SchemaSpec
) -> dict[str, dict[str, list[tuple[str, list[str]]]]]:
    """product -> field_key -> [(source_url, values)]"""
    out: dict[str, dict[str, list[tuple[str, list[str]]]]] = {}
    for ext in extractions:
        if ext.error:
            continue
        bucket = out.setdefault(ext.product, {})
        for key, claim in ext.claims.items():
            if claim.found and claim.values:
                bucket.setdefault(key, []).append((ext.url, claim.values))
    return out


# ---------------------------------------------------------------------- mechanical


def _mechanical(
    field: FieldSpec, claims: list[tuple[str, list[str]]], spec: SchemaSpec
) -> ReconciledField:
    sources = [u for u, _ in claims]
    flat: list[str] = []
    for _, vals in claims:
        flat.extend(vals)

    rf = ReconciledField(
        field_key=field.key,
        sources=sources,
        source_count=len(claims),
        method="single" if len(claims) == 1 else "mechanical",
    )

    if field.is_list:
        items = _dedupe_points(flat, field.max_items)
        rf.items = items
        rf.value = field.render(items, spec.list_output)
        return rf

    # Scalar field: pick the best-supported value, and flag real disagreement.
    groups: dict[str, list[str]] = {}
    for v in flat:
        groups.setdefault(_norm(v), []).append(v)
    ranked = sorted(
        groups.items(), key=lambda kv: (-len(kv[1]), -len(max(kv[1], key=len)))
    )
    best = max(ranked[0][1], key=len) if ranked else ""
    rf.items = [best] if best else []
    rf.value = best
    if len(ranked) > 1:
        rf.conflict = True
        alts = [max(v, key=len) for _, v in ranked[1:]]
        rf.conflict_note = "Sources disagree: " + " | ".join([best] + alts[:3])
    return rf


# ----------------------------------------------------------------------- llm merge


def _merge_schema(fields: list[FieldSpec]) -> dict:
    props = {}
    for f in fields:
        props[f.key] = {
            "type": "object",
            "description": f.label,
            "properties": {
                "value": f.json_value_property(),
                "conflict": {
                    "type": "boolean",
                    "description": "True only if sources state incompatible facts.",
                },
                "conflict_note": {
                    "type": "string",
                    "description": "Explain the contradiction, naming the differing "
                    "values. Empty when conflict is false.",
                },
            },
            "required": ["value", "conflict", "conflict_note"],
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


def _merge_prompt(
    spec: SchemaSpec,
    product: str,
    batch: list[tuple[FieldSpec, list[tuple[str, list[str]]]]],
) -> str:
    lines = [f"{spec.entity_label.upper()}: {product}", ""]
    for field, claims in batch:
        shape = {
            "list": f"list of up to {field.max_items} distinct points",
            "short_text": "one short phrase",
            "prose": "2-4 sentences",
        }[field.shape.value]
        lines += [
            f"### FIELD {field.key} ({field.label}) [{shape}]",
            f"Question: {field.question.strip()}",
        ]
        if field.guidance.strip():
            lines.append(f"Guidance: {field.guidance.strip()}")
        if field.anchors.strip():
            lines.append(f"Must cover: {field.anchors.strip()}")
        lines.append("Claims from sources:")
        for i, (url, vals) in enumerate(claims, 1):
            lines.append(f"  [{i}] source: {url}")
            for v in vals:
                lines.append(f"      - {v}")
        lines.append("")
    lines.append(
        f"Consolidate each field into one entry describing {product}. Merge duplicate "
        "points, keep all distinct ones, and flag only genuine factual contradictions."
    )
    return "\n".join(lines)


def _llm_merge_batch(
    spec: SchemaSpec,
    product: str,
    batch: list[tuple[FieldSpec, list[tuple[str, list[str]]]]],
    provider: BaseProvider,
    cfg: Config,
    cache: DiskCache,
    usage: Usage,
) -> dict[str, ReconciledField]:
    fields = [f for f, _ in batch]
    system = merge_system_prompt(spec, cfg)
    payload_fingerprint = make_key(*[f"{f.key}:{c}" for f, c in batch])
    ck = make_key(
        "reconcile",
        RECONCILE_CACHE_VERSION,
        provider.name,
        provider.model,
        cfg.temperature,
        system,
        product,
        payload_fingerprint,
    )

    data = cache.get(ck)
    if data is None:
        try:
            resp = provider.complete_json(
                system,
                _merge_prompt(spec, product, batch),
                _merge_schema(fields),
                max_tokens=cfg.max_output_tokens,
                temperature=cfg.temperature,
            )
        except Exception as e:  # noqa: BLE001
            usage.errors += 1
            # Degrade to mechanical rather than losing the batch entirely.
            out = {f.key: _mechanical(f, claims, spec) for f, claims in batch}
            for rf in out.values():
                rf.conflict_note = (
                    rf.conflict_note + " | " if rf.conflict_note else ""
                ) + f"LLM merge failed ({type(e).__name__}); merged mechanically."
            return out
        data = resp.data
        cache.set(ck, data)
        usage.calls += 1
        usage.prompt_tokens += resp.prompt_tokens
        usage.completion_tokens += resp.completion_tokens
    else:
        usage.cached += 1

    payload = (data or {}).get("fields") or {}
    out: dict[str, ReconciledField] = {}
    for field, claims in batch:
        node = payload.get(field.key) or {}
        raw = node.get("value")
        if isinstance(raw, list):
            items = [str(x).strip() for x in raw if str(x).strip()]
        else:
            items = [str(raw).strip()] if str(raw or "").strip() else []
        if field.is_list:
            items = _dedupe_points(items, field.max_items)

        rf = ReconciledField(
            field_key=field.key,
            items=items,
            value=field.render(items, spec.list_output),
            sources=[u for u, _ in claims],
            source_count=len(claims),
            conflict=bool(node.get("conflict")),
            conflict_note=str(node.get("conflict_note") or "").strip(),
            method="llm",
        )
        # Guard: the LLM sometimes returns the product name as the field
        # value (e.g. "n8n" for a "Key capabilities" field).  Detect and
        # fall back to mechanical — the product name is never a valid answer.
        value_stripped = rf.value.strip().lower()
        product_stripped = product.strip().lower()
        if value_stripped == product_stripped or (
            rf.items
            and len(rf.items) == 1
            and rf.items[0].strip().lower() == product_stripped
        ):
            rf = _mechanical(field, claims, spec)
            rf.method = "mechanical"
            note = "LLM returned product name as value; merged mechanically."
            rf.conflict_note = (
                f"{rf.conflict_note} | {note}" if rf.conflict_note else note
            )

        if not rf.value and not rf.items:
            rf = _mechanical(field, claims, spec)
            rf.method = "mechanical"
            note = "LLM merge returned nothing; merged mechanically."
            rf.conflict_note = (
                f"{rf.conflict_note} | {note}" if rf.conflict_note else note
            )
        out[field.key] = rf
    return out


# ------------------------------------------------------------------ web arbitration


def _web_check_schema(field: FieldSpec) -> dict:
    return {
        "type": "object",
        "properties": {
            "resolved": {
                "type": "boolean",
                "description": "True only if search established the correct value.",
            },
            "value": field.json_value_property(),
            "reasoning": {
                "type": "string",
                "description": "What you found and which source settled it.",
            },
        },
        "required": ["resolved", "value", "reasoning"],
        "additionalProperties": False,
    }


def _web_check_prompt(
    spec: SchemaSpec,
    product: str,
    field: FieldSpec,
    rf: ReconciledField,
    claims: list[tuple[str, list[str]]],
) -> str:
    shape = {
        "list": f"a list of up to {field.max_items} distinct points",
        "short_text": "one short phrase",
        "prose": "2-4 sentences",
    }[field.shape.value]

    lines = [
        f"{spec.entity_label.upper()}: {product}",
        f"FIELD: {field.label} ({field.key})",
        f"QUESTION: {field.question.strip()}",
    ]
    if field.guidance.strip():
        lines.append(f"GUIDANCE: {field.guidance.strip()}")
    if field.anchors.strip():
        lines.append(f"MUST COVER: {field.anchors.strip()}")
    lines += [f"ANSWER SHAPE: {shape}", "", "CONFLICTING CLAIMS FROM SCRAPED SOURCES:"]
    for i, (url, vals) in enumerate(claims, 1):
        lines.append(f"  [{i}] {url}")
        for v in vals:
            lines.append(f"      - {v}")
    if rf.conflict_note:
        lines += ["", f"WHY THIS WAS FLAGGED: {rf.conflict_note}"]
    lines += [
        "",
        f"Search the web and determine the correct current answer for {product}. "
        "Check the vendor's own site first.",
    ]
    return "\n".join(lines)


def _web_check_field(
    spec: SchemaSpec,
    product: str,
    field: FieldSpec,
    rf: ReconciledField,
    claims: list[tuple[str, list[str]]],
    provider: BaseProvider,
    cfg: Config,
    cache: DiskCache,
) -> WebVerdict:
    """Arbitrate one conflicting field. Never raises: failures become a noted verdict."""
    system = web_system_prompt(spec, cfg)
    ck = make_key(
        "webcheck",
        WEB_CHECK_CACHE_VERSION,
        provider.name,
        provider.model,
        cfg.web_check_max_searches,
        system,
        product,
        field.key,
        rf.value,
        rf.conflict_note,
        *[f"{u}:{v}" for u, v in claims],
    )
    cached = cache.get(ck)
    if cached is not None:
        try:
            verdict = WebVerdict.model_validate(cached)
            # Cached tokens and searches were already paid for on the original run.
            verdict.prompt_tokens = verdict.completion_tokens = verdict.searches = 0
            verdict.from_cache = True
            return verdict
        except Exception:  # noqa: BLE001 - stale cache shape, just re-run
            pass

    try:
        resp = provider.search_json(
            system,
            _web_check_prompt(spec, product, field, rf, claims),
            _web_check_schema(field),
            max_tokens=cfg.max_output_tokens,
            max_searches=cfg.web_check_max_searches,
        )
    except Exception as e:  # noqa: BLE001 - surfaced on the field, run continues
        return WebVerdict(resolved=False, error=f"{type(e).__name__}: {e}")

    data = resp.data or {}
    raw = data.get("value")
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
    else:
        items = [str(raw).strip()] if str(raw or "").strip() else []
    if field.is_list:
        items = _dedupe_points(items, field.max_items)

    verdict = WebVerdict(
        # A "resolved" verdict with no value is not usable, so treat it as unresolved.
        resolved=bool(data.get("resolved")) and bool(items),
        value=field.render(items, spec.list_output),
        items=items,
        reasoning=str(data.get("reasoning") or "").strip(),
        citations=resp.citations,
        searches=resp.searches,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
    )
    cache.set(ck, verdict.model_dump())
    return verdict


def _apply_verdict(rf: ReconciledField, verdict: WebVerdict) -> None:
    """Fold a verdict into the field, keeping the conflict visible either way."""
    rf.web = verdict
    if not verdict.resolved:
        note = (
            f"Web check failed ({verdict.error})"
            if verdict.error
            else "Web check could not settle this"
        )
        detail = f": {verdict.reasoning}" if verdict.reasoning else ""
        rf.conflict_note = f"{rf.conflict_note} | {note}{detail}".strip(" |")
        return

    rf.value_before_web = rf.value
    rf.value, rf.items = verdict.value, verdict.items
    rf.method = "web"
    # The conflict flag stays on: the sources really did disagree, and a reviewer should
    # still see that this cell was arbitrated rather than agreed.
    cites = ", ".join(verdict.citations[:3]) or "no citations returned"
    rf.conflict_note = (
        f"{rf.conflict_note} | Resolved by web check: {verdict.reasoning} "
        f"[was: {rf.value_before_web or '(empty)'}] Sources: {cites}"
    ).strip(" |")


def web_resolve_conflicts(
    results: list[ProductResult],
    extractions: list[SourceExtraction],
    spec: SchemaSpec,
    cfg: Config,
    provider: BaseProvider,
    progress: ProgressCb = None,
) -> Usage:
    """Second pass over conflicting fields, checking each against live web sources."""
    usage = Usage()
    if not provider.supports_search:
        return usage

    cache = DiskCache("webcheck", enabled=cfg.use_cache)
    grouped = _collect(extractions, spec)
    by_key = {f.key: f for f in spec.fields}

    jobs: list[tuple[ProductResult, FieldSpec, ReconciledField, list]] = []
    for r in results:
        for key, rf in r.fields.items():
            field = by_key.get(key)
            if rf.conflict and field is not None and field.fill_from.value != "entity":
                jobs.append((r, field, rf, grouped.get(r.product, {}).get(key, [])))

    total = len(jobs)
    if not total:
        if progress:
            progress(1, 1, "no conflicts to check")
        return usage

    done = 0
    if progress:
        progress(0, total, f"checking {total} conflicting field(s) against the web")

    # Searches are slow and network-bound, so they overlap; the vendor rate-limits them.
    workers = max(1, min(cfg.llm_concurrency, total))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _web_check_field,
                spec,
                r.product,
                field,
                rf,
                claims,
                provider,
                cfg,
                cache,
            ): (r, field, rf)
            for r, field, rf, claims in jobs
        }
        for fut in as_completed(futures):
            r, field, rf = futures[fut]
            try:
                verdict = fut.result()
            except Exception as e:  # noqa: BLE001 - defensive; _web_check_field catches
                verdict = WebVerdict(resolved=False, error=f"{type(e).__name__}: {e}")

            _apply_verdict(rf, verdict)
            if verdict.error:
                usage.errors += 1
            elif verdict.from_cache:
                usage.cached += 1
            else:
                usage.calls += 1
            usage.prompt_tokens += verdict.prompt_tokens
            usage.completion_tokens += verdict.completion_tokens
            usage.searches += verdict.searches

            done += 1
            if progress:
                state = (
                    "resolved"
                    if verdict.resolved
                    else ("failed" if verdict.error else "unresolved")
                )
                progress(done, total, f"{r.product}: {field.key} -- {state}")

    return usage


# -------------------------------------------------------------------------- public


def reconcile(
    extractions: list[SourceExtraction],
    spec: SchemaSpec,
    cfg: Config,
    provider: BaseProvider | None = None,
    use_llm: bool = True,
    progress: ProgressCb = None,
) -> tuple[list[ProductResult], Usage]:
    cache = DiskCache("reconcile", enabled=cfg.use_cache)
    usage = Usage()
    grouped = _collect(extractions, spec)

    products = sorted({e.product for e in extractions})
    results: list[ProductResult] = []

    total_steps = max(1, len(products))
    for pi, product in enumerate(products, 1):
        by_field = grouped.get(product, {})
        pr = ProductResult(
            product=product,
            pages_used=sorted(
                {e.url for e in extractions if e.product == product and not e.error}
            ),
            pages_failed=sorted(
                {e.url for e in extractions if e.product == product and e.error}
            ),
        )

        needs_llm: list[tuple[FieldSpec, list[tuple[str, list[str]]]]] = []

        for field in spec.fields:
            # Fields taken straight from the input sheet cost nothing.
            if field.fill_from.value == "entity":
                pr.fields[field.key] = ReconciledField(
                    field_key=field.key, value=product, items=[product], method="entity"
                )
                continue

            claims = by_field.get(field.key, [])
            if not claims:
                pr.fields[field.key] = ReconciledField(
                    field_key=field.key, method="empty"
                )
            elif len(claims) == 1 or not use_llm or provider is None:
                pr.fields[field.key] = _mechanical(field, claims, spec)
            else:
                needs_llm.append((field, claims))

        size = max(1, cfg.reconcile_batch_size)
        for start in range(0, len(needs_llm), size):
            batch = needs_llm[start : start + size]
            if progress:
                progress(
                    pi,
                    total_steps,
                    f"{product}: merging {', '.join(f.key for f, _ in batch)}",
                )
            pr.fields.update(
                _llm_merge_batch(spec, product, batch, provider, cfg, cache, usage)
            )

        if progress:
            progress(pi, total_steps, f"{product}: done")
        results.append(pr)

    return results, usage
