"""Import the client's "data structure" workbook into a SchemaSpec.

The brief arrives as a spreadsheet whose columns are an article outline:

    Article - H2 | Article - H3 | Key | Prompt | Anchors (tool-specific) | Source urls |
    Custom input

Three quirks drive the parsing:

* **H2 and H3 are merged cells written once.** A blank H2 means "same as the row
  above", so both are forward-filled or every field after the first loses its section.
* **One sheet per scenario.** The "General" sheet covers the article's subject product;
  the "Tools" sheet covers each alternative. They need different system prompts, so the
  scenario is taken from the sheet name and stored on the schema.
* **Source urls are many-to-many.** One key lists several URLs and the same URL recurs
  under several keys. That mapping *is* the routing: a URL is fetched once and asked
  only the questions it was nominated for.
"""

from __future__ import annotations

import io
import re

import pandas as pd

from pipeline.schema import FieldShape, FieldSpec, Scenario, SchemaSpec

COLUMN_ALIASES = {
    "section": {"article - h2", "article h2", "h2", "section"},
    "label": {"article - h3", "article h3", "h3", "subsection"},
    "key": {"key", "field", "field key"},
    "question": {"prompt", "question", "instruction"},
    "anchors": {
        "anchors (tool-specific)",
        "anchors tool specific",
        "anchors",
        "anchor",
    },
    "source_urls": {"source urls", "source url", "urls", "url", "sources"},
    "custom_input": {"custom input", "custom", "notes", "analyst input"},
}

URL_RE = re.compile(r"https?://\S+")

# A prompt asking for "categories", "examples" or "key events" wants a list; one asking
# to "explain" wants prose. Guessing here saves the user editing 16 rows by hand, and
# the Schema tab makes any wrong guess a one-click fix.
_LIST_HINTS = (
    "list them",
    "categories",
    "examples",
    "key events",
    "key strengths",
    "key limitations",
)
_SHORT_HINTS = (
    "minimum cost",
    "specifically whether",
    "specifically the metric",
    "specifically the distribution",
)


def _norm_header(h: str) -> str:
    t = re.sub(r"[\u2010-\u2015]", "-", str(h or "").strip().lower())
    return re.sub(r"\s+", " ", t).strip()


def _map_columns(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for col in df.columns:
        norm = _norm_header(col)
        for target, aliases in COLUMN_ALIASES.items():
            if target not in out and norm in aliases:
                out[target] = col
                break
    return out


def _guess_shape(question: str, key: str) -> FieldShape:
    q = question.lower()
    if any(h in q for h in _SHORT_HINTS):
        return FieldShape.SHORT_TEXT
    if any(h in q for h in _LIST_HINTS):
        return FieldShape.LIST
    if key.endswith(("_examples", "_limitations", "_strengths", "_automations")):
        return FieldShape.LIST
    return FieldShape.PROSE


def _scenario_for(sheet_name: str) -> Scenario:
    return Scenario.TOOLS if "tool" in sheet_name.lower() else Scenario.GENERAL


def _slug(text: str) -> str:
    t = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    if not t or not t[0].isalpha():
        t = f"f_{t}" if t else "field"
    return t


def _read_sheet(data: bytes, sheet: str) -> pd.DataFrame:
    """Header row is not always row 0 -- some exports carry a title row above it."""
    raw = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None, dtype=str)
    raw = raw.fillna("")
    header_row = 0
    for i in range(min(5, len(raw))):
        if any(_norm_header(v) in COLUMN_ALIASES["key"] for v in raw.iloc[i]):
            header_row = i
            break
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = [str(v).strip() for v in raw.iloc[header_row]]
    return df.reset_index(drop=True)


def list_sheets(data: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


def parse_brief_sheet(
    data: bytes,
    sheet: str,
    name: str = "",
    entity_label: str = "Product",
) -> tuple[SchemaSpec, list[str]]:
    """Turn one sheet of the brief workbook into a SchemaSpec.

    Returns the spec plus warnings about rows that could not be used.
    """
    warnings: list[str] = []
    df = _read_sheet(data, sheet)
    cols = _map_columns(df)

    if "key" not in cols and "label" not in cols:
        raise ValueError(
            f"Sheet {sheet!r} has neither a 'Key' nor an 'Article - H3' column, so "
            "there is nothing to build fields from."
        )

    def cell(rec, target: str) -> str:
        col = cols.get(target)
        return str(rec[col]).strip() if col else ""

    fields: list[FieldSpec] = []
    section = ""
    h3 = ""
    seen_keys: set[str] = set()
    seen_labels: set[str] = set()

    for i, rec in df.iterrows():
        # Merged cells: a blank inherits the last non-blank above it.
        section = cell(rec, "section") or section
        h3 = cell(rec, "label") or h3

        key = _slug(cell(rec, "key") or h3)
        question = cell(rec, "question")
        if not question:
            if cell(rec, "key") or cell(rec, "label"):
                warnings.append(f"{sheet} row {i + 2}: no Prompt, row skipped.")
            continue
        if key in seen_keys:
            warnings.append(f"{sheet} row {i + 2}: duplicate key {key!r}, row skipped.")
            continue
        seen_keys.add(key)

        # An H3 spanning several rows would give every one of them the same label, and
        # the label is the exported column header, so it has to stay unique.
        label = h3 or key.replace("_", " ").title()
        if label in seen_labels:
            label = f"{h3} — {key.replace('_', ' ')}"
        seen_labels.add(label)

        # The same URL is listed under several keys and one key lists several URLs, so
        # the key->URL map is the routing mechanism; nodes are not derived.
        urls: list[str] = []
        for u in URL_RE.findall(cell(rec, "source_urls")):
            if u not in urls:
                urls.append(u)

        fields.append(
            FieldSpec(
                key=key,
                label=label,
                question=question,
                shape=_guess_shape(question, key),
                max_items=10,
                anchors=cell(rec, "anchors"),
                custom_input=cell(rec, "custom_input"),
                section=section,
                source_urls=urls,
            )
        )

    if not fields:
        raise ValueError(f"Sheet {sheet!r} produced no usable fields.")

    scenario = _scenario_for(sheet)
    spec = SchemaSpec(
        name=_slug(name or sheet),
        entity_label=entity_label,
        description=f"Imported from workbook sheet {sheet!r}.",
        scenario=scenario,
        fields=fields,
    )
    return spec, warnings
