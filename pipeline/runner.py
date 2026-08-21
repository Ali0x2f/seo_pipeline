"""Input parsing, preflight checks and stage orchestration."""

from __future__ import annotations

import csv
import io
import math
import re
from typing import Callable, Iterable

import pandas as pd

from config import Config
from pipeline.extractor import extract_pages
from pipeline.models import InputRow, RunState, custom_input_url
from pipeline.providers import BaseProvider, Usage
from pipeline.reconciler import reconcile, web_resolve_conflicts
from pipeline.schema import SchemaSpec
from pipeline.scraper import normalize_url, scrape_rows
from pipeline.store import new_run_id, save_run

URL_RE = re.compile(r"https?://", re.I)

PRODUCT_ALIASES = {
    "product",
    "products",
    "tool",
    "platform",
    "vendor",
    "entity",
    "subject",
    "alternative",
    "alternative name",
    "name",
    "company",
    "app",
}
NODE_ALIASES = {
    "node",
    "nodes",
    "category",
    "categories",
    "section",
    "field",
    "fields",
    "topic",
    "group",
    "type",
    "aspect",
    "dimension",
}
URL_ALIASES = {
    "url",
    "urls",
    "link",
    "links",
    "uri",
    "source",
    "source url",
    "source urls",
    "page",
    "address",
    "href",
    "reference",
}
KEY_ALIASES = {
    "key",
    "keys",
    "field key",
    "field keys",
    "fields",
}

# A cell may list several keys; the brief writes them one per line.
KEY_SPLIT_RE = re.compile(r"[\n,;|]+")


def _norm_header(h: str) -> str:
    return re.sub(r"\s+", " ", str(h or "").strip().lower()).strip()


def _read_table(data: bytes | str, filename: str = "") -> pd.DataFrame:
    """Read CSV / TSV / Excel / pasted text into a DataFrame of strings."""
    name = (filename or "").lower()

    if name.endswith((".xlsx", ".xlsm", ".xls")) and isinstance(data, bytes):
        return pd.read_excel(io.BytesIO(data), dtype=str).fillna("")

    text = (
        data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
    )
    text = text.replace("\r\n", "\n").strip("\n")
    if not text.strip():
        raise ValueError("Input is empty.")

    sample = "\n".join(text.splitlines()[:20])
    if name.endswith(".tsv") or sample.count("\t") >= sample.count(","):
        delim = "\t"
    else:
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delim = ","

    return pd.read_csv(
        io.StringIO(text),
        sep=delim,
        dtype=str,
        engine="python",
        skip_blank_lines=True,
    ).fillna("")


def _pick_url_column(df: pd.DataFrame) -> str | None:
    best, best_score = None, 0
    for col in df.columns:
        score = sum(1 for v in df[col] if URL_RE.search(str(v)))
        if score > best_score:
            best, best_score = col, score
    return best if best_score else None


def parse_input(
    data: bytes | str,
    filename: str = "",
    default_product: str = "",
) -> tuple[list[InputRow], list[str]]:
    """Parse a URL sheet into InputRows.

    Column names are matched loosely, and the URL column is detected by content when the
    header is unrecognised -- pasting straight from a spreadsheet should just work.
    """
    warnings: list[str] = []
    df = _read_table(data, filename)
    if df.empty:
        raise ValueError("Input contains no data rows.")

    headers = {c: _norm_header(c) for c in df.columns}
    product_col = node_col = url_col = key_col = None
    # Each role is matched independently: a header can belong to more than one alias set
    # ("fields" is both a key and a node name), and a second URL-ish column must not be
    # mistaken for one of the others.
    for col, norm in headers.items():
        if url_col is None and norm in URL_ALIASES:
            url_col = col
            continue
        if key_col is None and norm in KEY_ALIASES:
            key_col = col
            continue
        if node_col is None and norm in NODE_ALIASES:
            node_col = col
            continue
        if product_col is None and norm in PRODUCT_ALIASES:
            product_col = col

    if url_col is None:
        url_col = _pick_url_column(df)
        if url_col is None:
            raise ValueError(
                "No URL column found. Include a column named 'URL' (or one whose values "
                "start with http)."
            )
        warnings.append(f"Using column {url_col!r} as the URL column.")

    # Neither Keys nor Node is required: a sheet of bare URLs is routed by the schema's
    # own key->URL map, and failing that every field is attempted.
    if key_col is None and node_col is None:
        warnings.append(
            "No Keys or Node column; each URL is routed using the schema's own source "
            "URLs, or tried against every field if it is not listed there."
        )

    if product_col is None and not default_product:
        raise ValueError(
            "No Product column found and no default product name given. Either add a "
            "'Product' column or set a default product name."
        )
    if product_col is None:
        warnings.append(
            f"No Product column; assigning every row to {default_product!r}."
        )

    # The brief lists the same URL under several keys, so rows are merged per
    # (product, url): one fetch, one extraction call, covering every key it was
    # nominated for.
    merged: dict[tuple[str, str], InputRow] = {}
    order: list[tuple[str, str]] = []
    for i, rec in df.iterrows():
        url = str(rec[url_col]).strip()
        if not url or not URL_RE.search(url):
            if url:
                warnings.append(f"Row {i + 2}: skipped, not a URL: {url[:60]!r}")
            continue
        node = str(rec[node_col]).strip() if node_col else ""
        product = (
            str(rec[product_col]).strip() if product_col else ""
        ) or default_product
        if not product:
            warnings.append(f"Row {i + 2}: skipped, no product name.")
            continue

        keys = (
            [k.strip() for k in KEY_SPLIT_RE.split(str(rec[key_col])) if k.strip()]
            if key_col
            else []
        )

        dedupe_key = (product.lower(), normalize_url(url))
        existing = merged.get(dedupe_key)
        if existing is None:
            merged[dedupe_key] = InputRow(
                product=product, node=node, url=url, keys=keys
            )
            order.append(dedupe_key)
            continue
        for k in keys:
            if k not in existing.keys:
                existing.keys.append(k)
        existing.node = existing.node or node

    rows = [merged[k] for k in order]
    if not rows:
        raise ValueError("No usable rows found in the input.")
    return rows, warnings


def custom_input_rows(spec: SchemaSpec, rows: list[InputRow]) -> list[InputRow]:
    """One extra source per field carrying analyst-pasted evidence.

    The client's brief supplies text for fields where scraping is unwanted (a Reddit
    reply, a note from a call). Turning it into a row means it flows through extraction,
    merging and provenance exactly like a fetched page, instead of being a special case
    bolted onto the prompt.
    """
    products = sorted({r.product for r in rows}) or [""]
    existing = {r.url for r in rows}
    out: list[InputRow] = []
    for f in spec.custom_input_fields():
        url = custom_input_url(f.key)
        if url in existing:
            continue
        for p in products:
            out.append(
                InputRow(
                    product=p,
                    url=url,
                    keys=[f.key],
                    custom_text=f.custom_input.strip(),
                )
            )
    return out


# ------------------------------------------------------------------------ preflight


def preflight(rows: list[InputRow], spec: SchemaSpec, cfg: Config) -> dict:
    """Report what the run will do, and what it cannot do, before spending money."""
    problems: list[str] = []
    notes: list[str] = []

    unknown: dict[str, int] = {}
    known_keys = set(spec.field_keys)
    for r in rows:
        # Only warn about node spelling when the row actually relies on node routing.
        if r.node and not r.keys and not spec.has_url_map:
            _, warn = spec.resolve_node(r.node)
            if warn:
                unknown[warn] = unknown.get(warn, 0) + 1
        for k in r.keys:
            if k not in known_keys:
                problems.append(
                    f"Row {r.url[:50]} names key {k!r}, which is not in this schema."
                )
        if not spec.fields_for(keys=r.keys, node=r.node, url=r.url):
            problems.append(
                f"{r.url[:50]} maps to no field, so it will be fetched but produce "
                "nothing."
            )
    problems.extend(unknown.keys())

    products = sorted({r.product for r in rows})
    covered: dict[str, set[str]] = {p: set() for p in products}
    for r in rows:
        for f in spec.fields_for(keys=r.keys, node=r.node, url=r.url):
            covered[r.product].add(f.key)
    # Custom input is injected at run time, so it counts as coverage here too.
    custom_keys = {f.key for f in spec.custom_input_fields()}
    for p in products:
        covered[p] |= custom_keys

    extractable = {f.key for f in spec.fields if f.fill_from.value == "extract"}
    for p in products:
        missing = extractable - covered[p]
        if missing:
            labels = [f.label for f in spec.fields if f.key in missing]
            problems.append(
                f"{p}: no URL feeds {len(missing)} field(s) -> "
                f"{', '.join(labels[:6])}{'...' if len(labels) > 6 else ''}. "
                "These will be blank."
            )

    est_extract_calls = sum(
        1 for r in rows if spec.fields_for(keys=r.keys, node=r.node, url=r.url)
    )
    fields_per_product = len(extractable)
    est_reconcile_calls = len(products) * math.ceil(
        max(1, fields_per_product) / max(1, cfg.reconcile_batch_size)
    )
    est_prompt_tokens = int(est_extract_calls * cfg.max_scrape_chars * 0.3 / 4)

    notes.append(
        f"{len(rows)} URLs across {len(products)} product(s); about "
        f"{est_extract_calls} extraction call(s) and up to {est_reconcile_calls} "
        "merge call(s)."
    )

    return {
        "problems": sorted(set(problems)),
        "notes": notes,
        "products": products,
        "urls": len(rows),
        "est_extract_calls": est_extract_calls,
        "est_reconcile_calls": est_reconcile_calls,
        "est_prompt_tokens": est_prompt_tokens,
        "lint": spec.lint(),
    }


# ---------------------------------------------------------------------- orchestration

StageCb = Callable[[str, int, int, str], None] | None


def run_pipeline(
    rows: Iterable[InputRow],
    spec: SchemaSpec,
    cfg: Config,
    provider: BaseProvider,
    use_llm_merge: bool = True,
    state: RunState | None = None,
    on_progress: StageCb = None,
    stages: tuple[str, ...] = ("scrape", "extract", "reconcile"),
) -> tuple[RunState, dict[str, Usage]]:
    """Run the pipeline, persisting state after every stage.

    `stages` lets the UI re-run just the tail of the pipeline -- for example redoing
    extraction after a prompt change without re-fetching pages. An empty `stages` with a
    completed `state` runs nothing but the web conflict check.
    """
    rows = list(rows)
    rows += custom_input_rows(spec, rows)
    if state is None:
        state = RunState(run_id=new_run_id())
    state.inputs = rows
    state.schema_name = spec.name
    state.model = provider.model
    state.provider = provider.name

    def emit(stage: str):
        def cb(done: int, total: int, msg: str) -> None:
            if on_progress:
                on_progress(stage, done, total, msg)

        return cb

    usages: dict[str, Usage] = {}

    if "scrape" in stages or not state.pages:
        state.pages = scrape_rows(rows, cfg, progress=emit("scrape"))
        state.stage = "scraped"
        save_run(state)

    if "extract" in stages or not state.extractions:
        extractions, usage = extract_pages(
            state.pages, spec, provider, cfg, progress=emit("extract")
        )
        state.extractions = extractions
        usages["extract"] = usage
        state.stage = "extracted"
        save_run(state)

    if "reconcile" in stages or not state.results:
        results, usage = reconcile(
            state.extractions,
            spec,
            cfg,
            provider=provider,
            use_llm=use_llm_merge,
            progress=emit("reconcile"),
        )
        state.results = results
        usages["reconcile"] = usage
        state.stage = "reconciled"
        save_run(state)

    # Arbitration reads the reconciled results, so it runs last and can also be re-run on
    # its own against a loaded run without repeating the merge.
    if cfg.web_check_conflicts and provider.supports_search and state.results:
        usages["web_check"] = web_resolve_conflicts(
            state.results,
            state.extractions,
            spec,
            cfg,
            provider,
            progress=emit("web check"),
        )
        save_run(state)

    state.warnings = []
    for p in state.pages:
        if not p.success:
            state.warnings.append(f"scrape failed: {p.url} ({p.error})")
    for e in state.extractions:
        if e.error:
            state.warnings.append(f"extract failed: {e.url} ({e.error})")
    for r in state.results:
        for rf in r.fields.values():
            if rf.web and rf.web.error:
                state.warnings.append(
                    f"web check failed: {r.product}/{rf.field_key} ({rf.web.error})"
                )
    save_run(state)

    return state, usages
