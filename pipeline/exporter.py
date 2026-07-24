"""Stage 4: turn reconciled results into deliverables.

The dataset sheet is what the client pastes into their spreadsheet. The other sheets
exist so the numbers can be defended: every merged cell can be traced back to the
individual source claims and the quotes behind them.
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

from pipeline.models import ProductResult, ScrapedPage, SourceExtraction
from pipeline.schema import SchemaSpec

CONFLICT_MARK = "[CONFLICT] "


def build_dataset(
    results: list[ProductResult],
    spec: SchemaSpec,
    mark_conflicts: bool = True,
) -> pd.DataFrame:
    """One row per product, columns in schema order using the client's exact labels."""
    has_entity_field = bool(spec.entity_fields)
    rows: list[dict[str, str]] = []
    for r in results:
        row: dict[str, str] = {}
        if not has_entity_field:
            row[spec.entity_label] = r.product
        for f in spec.fields:
            rf = r.fields.get(f.key)
            val = rf.value if rf else ""
            if mark_conflicts and rf and rf.conflict and val:
                val = CONFLICT_MARK + val
            row[f.label] = val
        rows.append(row)

    cols = ([] if has_entity_field else [spec.entity_label]) + spec.labels
    return pd.DataFrame(rows, columns=cols)


def build_dataset_transposed(results: list[ProductResult], spec: SchemaSpec) -> pd.DataFrame:
    """Fields as rows, products as columns. Far easier to read for wide schemas."""
    data: dict[str, list[str]] = {"Field": spec.labels}
    for r in results:
        data[r.product] = [
            (r.fields.get(f.key).value if r.fields.get(f.key) else "") for f in spec.fields
        ]
    return pd.DataFrame(data)


def build_brief(spec: SchemaSpec) -> pd.DataFrame:
    """The schema itself as a table: what each column means and what feeds it."""
    return pd.DataFrame(
        [
            {
                "Field": f.label,
                "Key": f.key,
                "Shape": f.shape.value,
                "Source": f.fill_from.value,
                "Fed by nodes": ", ".join(f.nodes) if f.nodes else "(all nodes)",
                "Description": f.question.strip(),
                "Guidance": f.guidance.strip(),
            }
            for f in spec.fields
        ]
    )


def build_provenance(results: list[ProductResult], spec: SchemaSpec) -> pd.DataFrame:
    rows = []
    for r in results:
        for f in spec.fields:
            rf = r.fields.get(f.key)
            if rf is None:
                continue
            rows.append(
                {
                    "Product": r.product,
                    "Field": f.label,
                    "Value": rf.value,
                    "Sources used": len(rf.sources),
                    "Merge method": rf.method,
                    "Conflict": "YES" if rf.conflict else "",
                    "Conflict note": rf.conflict_note,
                    "Source URLs": "\n".join(rf.sources),
                }
            )
    return pd.DataFrame(rows)


def build_claims(
    extractions: list[SourceExtraction], spec: SchemaSpec
) -> pd.DataFrame:
    """Every individual claim with its supporting quote, for spot-checking."""
    label = {f.key: f.label for f in spec.fields}
    rows = []
    for e in extractions:
        for key, c in e.claims.items():
            if not c.found:
                continue
            rows.append(
                {
                    "Product": e.product,
                    "Field": label.get(key, key),
                    "Node": e.node,
                    "Source URL": e.url,
                    "Claim": " | ".join(c.values),
                    "Supporting quote": c.quote,
                    "Quote verified": "yes" if c.quote_verified else "NO",
                }
            )
    return pd.DataFrame(rows)


def build_pages(pages: list[ScrapedPage]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Product": p.product,
                "Node": p.node,
                "URL": p.url,
                "OK": "yes" if p.success else "NO",
                "Chars": p.char_count,
                "Method": p.fetch_method,
                "Truncated": "yes" if p.truncated else "",
                "Title": p.title,
                "Error": p.error or "",
            }
            for p in pages
        ]
    )


def build_coverage(results: list[ProductResult], spec: SchemaSpec) -> pd.DataFrame:
    """Which cells are empty and which need review. This is the QA view."""
    keys = spec.field_keys
    rows = []
    for r in results:
        empty = [
            f.label for f in spec.fields
            if not r.fields.get(f.key) or r.fields[f.key].is_empty
        ]
        conflicts = [
            f.label for f in spec.fields
            if r.fields.get(f.key) and r.fields[f.key].conflict
        ]
        single = [
            f.label for f in spec.fields
            if r.fields.get(f.key) and r.fields[f.key].source_count == 1
        ]
        rows.append(
            {
                "Product": r.product,
                "Coverage": f"{r.coverage(keys) * 100:.0f}%",
                "Filled": len(keys) - len(empty),
                "Empty": len(empty),
                "Conflicts": len(conflicts),
                "Single-source": len(single),
                "Pages used": len(r.pages_used),
                "Pages failed": len(r.pages_failed),
                "Empty fields": ", ".join(empty),
                "Conflicting fields": ", ".join(conflicts),
            }
        )
    return pd.DataFrame(rows)


def to_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = name[:31] or "Sheet"
            (df if not df.empty else pd.DataFrame({"(empty)": []})).to_excel(
                writer, index=False, sheet_name=safe
            )
            ws = writer.sheets[safe]
            for i, col in enumerate(df.columns, start=1):
                width = min(60, max(14, len(str(col)) + 4))
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
            ws.freeze_panes = "A2"
    return buf.getvalue()


def build_workbook(
    results: list[ProductResult],
    spec: SchemaSpec,
    extractions: list[SourceExtraction],
    pages: list[ScrapedPage],
) -> bytes:
    return to_excel(
        {
            "Dataset": build_dataset(results, spec),
            "Dataset (transposed)": build_dataset_transposed(results, spec),
            "QA coverage": build_coverage(results, spec),
            "Provenance": build_provenance(results, spec),
            "Claims": build_claims(extractions, spec),
            "Pages": build_pages(pages),
            "Brief": build_brief(spec),
        }
    )


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")
