"""Dynamic output-schema definition.

The whole point of this module: the shape of the deliverable is *data*, not code.
A content brief becomes a YAML file describing each column (its guiding question,
the shape of the answer, and which input node feeds it). Nothing here is specific
to any client or product category.
"""

from __future__ import annotations

import difflib
import re
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from config import SCHEMA_DIR

KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class FieldShape(str, Enum):
    SHORT_TEXT = "short_text"   # a phrase: "$20/mo", "Freemium + open-source"
    LIST = "list"               # discrete points, deduped and merged across sources
    PROSE = "prose"             # a paragraph or two of explanation


class ListOutput(str, Enum):
    SEMICOLON = "semicolon"
    BULLETS = "bullets"
    NEWLINE = "newline"


class FillFrom(str, Enum):
    EXTRACT = "extract"   # ask the LLM to find it in the source pages
    ENTITY = "entity"     # copy the product/entity name from the input sheet


def normalize_label(text: str) -> str:
    """Loose normalisation so 'Pricing ', 'pricing' and 'Pricing.' all match."""
    t = (text or "").strip().lower()
    t = t.replace("&", "and")
    t = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class FieldSpec(BaseModel):
    """One column of the output dataset."""

    key: str
    label: str
    question: str
    shape: FieldShape = FieldShape.PROSE
    nodes: list[str] = Field(default_factory=list)   # empty => fed by every node
    max_items: int = 10
    guidance: str = ""
    fill_from: FillFrom = FillFrom.EXTRACT

    @field_validator("key")
    @classmethod
    def _check_key(cls, v: str) -> str:
        v = v.strip()
        if not KEY_RE.match(v):
            raise ValueError(
                f"field key {v!r} must be snake_case, start with a letter, "
                "and contain only a-z, 0-9 and underscores"
            )
        return v

    @property
    def is_list(self) -> bool:
        return self.shape == FieldShape.LIST

    def prompt_line(self) -> str:
        bits = [self.question.strip()]
        if self.guidance.strip():
            bits.append(self.guidance.strip())
        if self.is_list:
            bits.append(f"Return up to {self.max_items} distinct points.")
        elif self.shape == FieldShape.SHORT_TEXT:
            bits.append("Answer in a short phrase, not a sentence.")
        else:
            bits.append("Answer in 2-4 sentences.")
        return " ".join(bits)

    def json_value_property(self) -> dict:
        if self.is_list:
            return {
                "type": "array",
                "items": {"type": "string"},
                "description": "One concise point per element. Empty array if absent.",
            }
        return {"type": "string", "description": "Empty string if absent."}

    def render(self, items: list[str], style: ListOutput) -> str:
        clean = [i.strip() for i in items if i and i.strip()]
        if not clean:
            return ""
        if not self.is_list:
            return clean[0] if len(clean) == 1 else " ".join(clean)
        if style == ListOutput.BULLETS:
            return "\n".join(f"- {c}" for c in clean)
        if style == ListOutput.NEWLINE:
            return "\n".join(clean)
        return "; ".join(c.rstrip(";").rstrip(".") if len(c) < 80 else c for c in clean)


class SchemaSpec(BaseModel):
    """A complete content brief: the columns to fill and the nodes that feed them."""

    name: str
    version: int = 1
    entity_label: str = "Product"
    description: str = ""
    nodes: list[str] = Field(default_factory=list)
    list_output: ListOutput = ListOutput.SEMICOLON
    fields: list[FieldSpec] = Field(default_factory=list)

    # ---------- basic accessors ----------

    @property
    def field_keys(self) -> list[str]:
        return [f.key for f in self.fields]

    @property
    def labels(self) -> list[str]:
        return [f.label for f in self.fields]

    def by_key(self, key: str) -> FieldSpec | None:
        for f in self.fields:
            if f.key == key:
                return f
        return None

    def all_nodes(self) -> list[str]:
        """Declared nodes, plus any referenced only by a field."""
        seen: list[str] = list(self.nodes)
        known = {normalize_label(n) for n in seen}
        for f in self.fields:
            for n in f.nodes:
                if normalize_label(n) not in known:
                    known.add(normalize_label(n))
                    seen.append(n)
        return seen

    # ---------- node resolution ----------

    def resolve_node(self, raw: str) -> tuple[str | None, str | None]:
        """Map a messy input node label onto a canonical one.

        Returns (canonical_node, warning). Exact-after-normalisation wins; otherwise
        we try a close match and say so, rather than silently mis-routing the row.
        """
        nodes = self.all_nodes()
        if not nodes:
            return None, None
        norm_map = {normalize_label(n): n for n in nodes}
        target = normalize_label(raw)
        if target in norm_map:
            return norm_map[target], None
        close = difflib.get_close_matches(target, list(norm_map), n=1, cutoff=0.8)
        if close:
            picked = norm_map[close[0]]
            return picked, f"Node {raw!r} interpreted as {picked!r}"
        return None, (
            f"Node {raw!r} matches nothing in schema {self.name!r}; "
            f"fields with no node restriction will still be attempted"
        )

    def fields_for_node(self, node: str | None) -> list[FieldSpec]:
        """Extractable fields fed by this node.

        A field with an empty `nodes` list is fed by every node. Fields sourced from
        the input sheet rather than the pages are excluded -- they cost nothing.
        """
        candidates = [f for f in self.fields if f.fill_from == FillFrom.EXTRACT]
        if node is None:
            return [f for f in candidates if not f.nodes]
        target = normalize_label(node)
        out = []
        for f in candidates:
            if not f.nodes:
                out.append(f)
            elif any(normalize_label(n) == target for n in f.nodes):
                out.append(f)
        return out

    @property
    def entity_fields(self) -> list[FieldSpec]:
        return [f for f in self.fields if f.fill_from == FillFrom.ENTITY]

    # ---------- validation ----------

    def lint(self) -> list[str]:
        """Non-fatal problems worth showing the user before a paid run."""
        problems: list[str] = []
        if not self.fields:
            problems.append("Schema has no fields.")

        seen: dict[str, int] = {}
        for f in self.fields:
            seen[f.key] = seen.get(f.key, 0) + 1
        for k, n in seen.items():
            if n > 1:
                problems.append(f"Duplicate field key {k!r} appears {n} times.")

        labels: dict[str, int] = {}
        for f in self.fields:
            labels[f.label] = labels.get(f.label, 0) + 1
        for k, n in labels.items():
            if n > 1:
                problems.append(f"Duplicate column label {k!r} appears {n} times.")

        declared = {normalize_label(n) for n in self.nodes}
        for f in self.fields:
            for n in f.nodes:
                if declared and normalize_label(n) not in declared:
                    problems.append(
                        f"Field {f.key!r} references node {n!r}, which is not in the "
                        "schema's `nodes` list."
                    )
        for n in self.nodes:
            if not self.fields_for_node(n):
                problems.append(f"Node {n!r} feeds no fields, so its URLs do nothing.")

        for f in self.fields:
            if not f.question.strip():
                problems.append(f"Field {f.key!r} has an empty question.")
        return problems

    # ---------- io ----------

    @classmethod
    def from_yaml_text(cls, text: str) -> SchemaSpec:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Schema YAML must be a mapping at the top level.")
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path) -> SchemaSpec:
        return cls.from_yaml_text(Path(path).read_text(encoding="utf-8"))

    def to_yaml_text(self) -> str:
        data = self.model_dump(mode="json", exclude_defaults=False)
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml_text(), encoding="utf-8")
        return p


def list_schemas() -> list[Path]:
    return sorted(SCHEMA_DIR.glob("*.yaml")) + sorted(SCHEMA_DIR.glob("*.yml"))


def load_schema_by_name(name: str) -> SchemaSpec:
    for p in list_schemas():
        if p.stem == name:
            return SchemaSpec.load(p)
    raise FileNotFoundError(f"No schema named {name!r} in {SCHEMA_DIR}")
