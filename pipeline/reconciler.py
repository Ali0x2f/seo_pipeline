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
"""

from __future__ import annotations

import re
from typing import Callable

from config import Config
from pipeline.cache import DiskCache, make_key
from pipeline.models import ProductResult, ReconciledField, SourceExtraction
from pipeline.providers import BaseProvider, Usage
from pipeline.schema import FieldSpec, SchemaSpec

RECONCILE_CACHE_VERSION = 2
ProgressCb = Callable[[int, int, str], None] | None

SYSTEM_PROMPT = (
    "You consolidate research notes into a single authoritative dataset entry.\n"
    "For each field you receive numbered claims taken from different source pages about "
    "the same product.\n"
    "Your job:\n"
    "1. Merge claims that make the same point, keeping the clearest and most specific "
    "wording. Never repeat the same point twice in different words.\n"
    "2. Preserve every distinct substantive point. Do not drop information just to be "
    "brief.\n"
    "3. Prefer concrete detail (figures, tier names, limits) over vague phrasing.\n"
    "4. If sources genuinely contradict each other on a fact, set conflict=true and "
    "explain the disagreement in conflict_note, naming the differing values. Use the "
    "best-supported value as the value. Differing levels of detail are NOT a conflict; "
    "only incompatible facts are.\n"
    "5. Introduce nothing that is not present in the claims. You have no other source.\n"
    "6. NEVER use the product name as the value for a field. The product name is just "
    "the subject — it is never a valid answer to a field's question. If no substantive "
    "claims exist for a field, return an empty value.\n"
    "7. Write in neutral, factual English with no marketing language."
)


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
    payload_fingerprint = make_key(*[f"{f.key}:{c}" for f, c in batch])
    ck = make_key(
        "reconcile",
        RECONCILE_CACHE_VERSION,
        provider.name,
        provider.model,
        cfg.temperature,
        product,
        payload_fingerprint,
    )

    data = cache.get(ck)
    if data is None:
        try:
            resp = provider.complete_json(
                SYSTEM_PROMPT,
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
