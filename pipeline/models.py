"""Domain models shared across the pipeline stages."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class InputRow(BaseModel):
    """One line of the user's input sheet."""

    product: str
    node: str
    url: str


class ScrapedPage(BaseModel):
    url: str
    node: str
    product: str
    success: bool = False
    title: str = ""
    text: str = ""
    char_count: int = 0
    fetch_method: str = ""  # crawl4ai | httpx-trafilatura | cache
    truncated: bool = False
    error: str | None = None
    fetched_at: str = Field(default_factory=_now)

    @property
    def is_thin(self) -> bool:
        return self.success and self.char_count < 600


class FieldClaim(BaseModel):
    """What a single source page says about a single schema field."""

    field_key: str
    found: bool = False
    values: list[str] = Field(default_factory=list)
    quote: str = ""
    # False when the model's supporting quote is not actually present in the page text,
    # which is the clearest available signal that a claim may be invented.
    quote_verified: bool = False

    @property
    def as_text(self) -> str:
        return "; ".join(v for v in self.values if v)


class SourceExtraction(BaseModel):
    """All claims harvested from one (product, url) pair."""

    product: str
    url: str
    node: str
    claims: dict[str, FieldClaim] = Field(default_factory=dict)
    model: str = ""
    error: str | None = None
    from_cache: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def found_count(self) -> int:
        return sum(1 for c in self.claims.values() if c.found)


class WebVerdict(BaseModel):
    """Outcome of checking one conflicting field against live web sources."""

    resolved: bool = False
    value: str = ""
    items: list[str] = Field(default_factory=list)
    reasoning: str = ""
    citations: list[str] = Field(default_factory=list)
    searches: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    checked_at: str = Field(default_factory=_now)
    from_cache: bool = False
    error: str = ""


class ReconciledField(BaseModel):
    """Final, merged value for one field of one product."""

    field_key: str
    value: str = ""
    items: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    source_count: int = 0
    conflict: bool = False
    conflict_note: str = ""
    method: str = ""  # empty | single | mechanical | llm | web
    # Set only when web arbitration ran on this field. The pre-arbitration value is kept
    # so a reviewer can see what the sources said before the web overrode them.
    web: WebVerdict | None = None
    value_before_web: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.value and not self.items


class ProductResult(BaseModel):
    product: str
    fields: dict[str, ReconciledField] = Field(default_factory=dict)
    pages_used: list[str] = Field(default_factory=list)
    pages_failed: list[str] = Field(default_factory=list)

    def coverage(self, field_keys: list[str]) -> float:
        if not field_keys:
            return 0.0
        filled = sum(
            1 for k in field_keys if k in self.fields and not self.fields[k].is_empty
        )
        return filled / len(field_keys)


class RunState(BaseModel):
    """Everything about one pipeline run, persisted to disk so work is never lost."""

    run_id: str
    schema_name: str = ""
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    model: str = ""
    provider: str = ""
    inputs: list[InputRow] = Field(default_factory=list)
    pages: list[ScrapedPage] = Field(default_factory=list)
    extractions: list[SourceExtraction] = Field(default_factory=list)
    results: list[ProductResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stage: str = "created"  # created|scraped|extracted|reconciled

    def touch(self) -> None:
        self.updated_at = _now()

    def page_for(self, product: str, url: str) -> ScrapedPage | None:
        for p in self.pages:
            if p.product == product and p.url == url:
                return p
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "inputs": len(self.inputs),
            "products": len({r.product for r in self.inputs}),
            "pages_ok": sum(1 for p in self.pages if p.success),
            "pages_failed": sum(1 for p in self.pages if not p.success),
            "extractions": len(self.extractions),
            "extraction_errors": sum(1 for e in self.extractions if e.error),
            "results": len(self.results),
        }
