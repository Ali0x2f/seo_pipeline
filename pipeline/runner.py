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
from pipeline.models import InputRow, RunState
from pipeline.providers import BaseProvider, Usage
from pipeline.reconciler import reconcile
from pipeline.schema import SchemaSpec
from pipeline.scraper import normalize_url, scrape_rows
from pipeline.store import new_run_id, save_run

URL_RE = re.compile(r"https?://", re.I)

PRODUCT_ALIASES = {
    "product", "products", "tool", "platform", "vendor", "entity", "subject",
    "alternative", "alternative name", "name", "company", "app",
}
NODE_ALIASES = {
    "node", "nodes", "category", "categories", "section", "field", "fields",
    "topic", "group", "type", "aspect", "dimension",
}
URL_ALIASES = {"url", "urls", "link", "links", "uri", "source", "source url", "page",
               "address", "href", "reference"}


def _norm_header(h: str) -> str:
    return re.sub(r"\s+", " ", str(h or "").strip().lower()).strip()


def _read_table(data: bytes | str, filename: str = "") -> pd.DataFrame:
    """Read CSV / TSV / Excel / pasted text into a DataFrame of strings."""
    name = (filename or "").lower()

    if name.endswith((".xlsx", ".xlsm", ".xls")) and isinstance(data, bytes):
        return pd.read_excel(io.BytesIO(data), dtype=str).fillna("")

    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
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
        io.StringIO(text), sep=delim, dtype=str, engine="python",
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
    product_col = node_col = url_col = None
    for col, norm in headers.items():
        if url_col is None and norm in URL_ALIASES:
            url_col = col
        elif node_col is None and norm in NODE_ALIASES:
            node_col = col
        elif product_col is None and norm in PRODUCT_ALIASES:
            product_col = col

    if url_col is None:
        url_col = _pick_url_column(df)
        if url_col is None:
            raise ValueError(
                "No URL column found. Include a column named 'URL' (or one whose values "
                "start with http)."
            )
        warnings.append(f"Using column {url_col!r} as the URL column.")

    if node_col is None:
        remaining = [c for c in df.columns if c not in (url_col, product_col)]
        if remaining:
            node_col = remaining[0]
            warnings.append(f"Using column {node_col!r} as the Node column.")
        else:
            raise ValueError(
                "No Node column found. Add a column naming the section each URL feeds."
            )

    if product_col is None and not default_product:
        raise ValueError(
            "No Product column found and no default product name given. Either add a "
            "'Product' column or set a default product name."
        )
    if product_col is None:
        warnings.append(f"No Product column; assigning every row to {default_product!r}.")

    rows: list[InputRow] = []
    seen: set[tuple[str, str, str]] = set()
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
        if not node:
            warnings.append(f"Row {i + 2}: skipped, no node.")
            continue

        dedupe_key = (product.lower(), node.lower(), normalize_url(url))
        if dedupe_key in seen:
            warnings.append(f"Row {i + 2}: duplicate of an earlier row, skipped.")
            continue
        seen.add(dedupe_key)
        rows.append(InputRow(product=product, node=node, url=url))

    if not rows:
        raise ValueError("No usable rows found in the input.")
    return rows, warnings


# ------------------------------------------------------------------------ preflight

def preflight(rows: list[InputRow], spec: SchemaSpec, cfg: Config) -> dict:
    """Report what the run will do, and what it cannot do, before spending money."""
    problems: list[str] = []
    notes: list[str] = []

    unknown: dict[str, int] = {}
    per_node_fields: dict[str, int] = {}
    for r in rows:
        canonical, warn = spec.resolve_node(r.node)
        if warn:
            unknown[warn] = unknown.get(warn, 0) + 1
        fields = spec.fields_for_node(canonical)
        per_node_fields[r.node] = len(fields)
        if not fields:
            problems.append(
                f"Node {r.node!r} feeds no fields, so its {r.url[:50]} will be scraped "
                "but produce nothing."
            )
    problems.extend(unknown.keys())

    products = sorted({r.product for r in rows})
    covered: dict[str, set[str]] = {p: set() for p in products}
    for r in rows:
        canonical, _ = spec.resolve_node(r.node)
        for f in spec.fields_for_node(canonical):
            covered[r.product].add(f.key)

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
        1 for r in rows if spec.fields_for_node(spec.resolve_node(r.node)[0])
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
    extraction after a prompt change without re-fetching pages.
    """
    rows = list(rows)
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
            state.extractions, spec, cfg,
            provider=provider, use_llm=use_llm_merge, progress=emit("reconcile"),
        )
        state.results = results
        usages["reconcile"] = usage
        state.stage = "reconciled"
        save_run(state)

    state.warnings = []
    for p in state.pages:
        if not p.success:
            state.warnings.append(f"scrape failed: {p.url} ({p.error})")
    for e in state.extractions:
        if e.error:
            state.warnings.append(f"extract failed: {e.url} ({e.error})")
    save_run(state)

    return state, usages
