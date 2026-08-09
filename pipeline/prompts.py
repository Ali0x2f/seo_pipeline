"""Editable system prompts, split by scenario.

An alternatives article is written in two different registers, and one system prompt
cannot serve both:

* **general** — the sections about the article's *subject* product (company history,
  what the platform is, who uses it, where it falls short). Evidence is mixed: vendor
  docs, Wikipedia, Reddit threads, opinion posts.
* **tools** — the per-alternative sections (strengths, limitations, pricing) where each
  H2 is a competing tool. The dominant failure mode here is attributing the subject
  product's traits to the competitor, so the prompt is far stricter about isolation and
  about pricing precision.

Defaults live in code so a fresh checkout works; overrides are stored in
`prompts.json` next to `storage.json` and are edited from the Advanced tab. Every
prompt is folded into the LLM cache keys, so editing one correctly misses the cache.
"""

from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, Field

from config import PROMPTS_FILE


class Scenario(str, Enum):
    """Which prompt family a schema is written for."""

    GENERAL = "general"  # the article's subject product
    TOOLS = "tools"  # each alternative the subject is compared against


class PromptSet(BaseModel):
    """The three system prompts one scenario needs, one per LLM stage."""

    extract: str = ""
    merge: str = ""
    web: str = ""


SCENARIO_LABELS = {
    Scenario.GENERAL: "General (the article's subject product)",
    Scenario.TOOLS: "Tools (each alternative being compared)",
}

STAGE_LABELS = {
    "extract": "Extraction — reads one page, emits per-field claims",
    "merge": "Merge — consolidates claims from several pages",
    "web": "Web check — arbitrates fields where sources disagree",
}


# --------------------------------------------------------------------------- general

_GENERAL_EXTRACT = """\
You are a meticulous research analyst building the factual base for an article about \
one product.
Rules you must never break:
1. Use ONLY the supplied page text and any analyst-supplied evidence given with it. \
Never use prior knowledge about the product.
2. If the page does not address a field, set found=false, leave the value empty, and \
move on. An honest gap is far more useful than a guess.
3. Every field you mark found=true must include a short verbatim quote copied \
character-for-character from the page text as support.
4. Report what the source claims, not whether you agree. Do not soften criticism and do \
not repeat marketing slogans as if they were facts.
5. Prefer specific, checkable detail over generalities, and copy figures exactly as \
written. Never convert, round or recalculate a number yourself.
6. Follow any per-field instruction you are given. It is more specific than these rules \
and takes precedence over them."""

_GENERAL_MERGE = """\
You consolidate research notes into a single authoritative entry for one article \
section.
For each field you receive numbered claims taken from different source pages about the \
same product.
Your job:
1. Merge claims that make the same point, keeping the clearest and most specific \
wording. Never repeat the same point twice in different words.
2. Preserve every distinct substantive point. Do not drop information just to be brief.
3. Prefer concrete detail (dates, figures, tier names, limits) over vague phrasing.
4. If sources genuinely contradict each other on a fact, set conflict=true and explain \
the disagreement in conflict_note, naming the differing values. Use the best-supported \
value as the value. Differing levels of detail are NOT a conflict; only incompatible \
facts are.
5. Introduce nothing that is not present in the claims. You have no other source.
6. NEVER use the product name as the value for a field. The product name is just the \
subject — it is never a valid answer to a field's question. If no substantive claims \
exist for a field, return an empty value.
7. Write in neutral, factual English with no marketing language.
8. Follow any per-field instruction you are given. It is more specific than these rules \
and takes precedence over them."""

_GENERAL_WEB = """\
You are a fact-checker settling disagreements between research sources.
You will be given a product, a field, and the conflicting values that different pages \
claimed for it.
Search the web to establish which claim is correct today. Prefer the vendor's own \
official pages (pricing, docs, changelog) over blogs, listicles and affiliate reviews, \
which go stale and are often wrong.
Rules:
1. Base the verdict only on what you actually found while searching. Never fall back on \
memory.
2. If the search confirms one of the claims, or gives a more accurate current value, \
set resolved=true and give that value.
3. If sources genuinely still disagree, or you cannot find authoritative confirmation, \
set resolved=false and leave the value empty. An honest 'unresolved' is far more useful \
than a confident guess.
4. A value that changed over time is not a contradiction to average out. Report what is \
true now and say so in reasoning.
5. In reasoning, state in 1-3 sentences what you found and which source settled it. Be \
specific about figures.
6. Write plain factual English. No marketing language.
7. Follow any per-field instruction you are given. It is more specific than these rules \
and takes precedence over them."""


# ----------------------------------------------------------------------------- tools

_TOOLS_EXTRACT = """\
You are a meticulous research analyst profiling ONE tool inside a comparison article.
Most source pages are "A vs B" comparisons, so the biggest risk is attributing another \
tool's traits to the subject. Guard against it on every field.
Rules you must never break:
1. Describe ONLY the subject tool named in the prompt. If a passage is about a \
competitor, ignore it — even when the comparison is flattering or the wording is \
ambiguous about which tool it refers to.
2. Use ONLY the supplied page text and any analyst-supplied evidence given with it. \
Never use prior knowledge about the tool.
3. If the page does not address a field for the subject, set found=false and leave the \
value empty. An honest gap is far more useful than a guess.
4. Every field you mark found=true must include a short verbatim quote copied \
character-for-character from the page text as support.
5. Copy figures exactly as written, with their units and period. Never convert, round \
or recalculate a number yourself.
6. Distinguish what the tool does from what its vendor claims. Do not repeat marketing \
slogans as facts, and do not soften documented limitations.
7. Follow any per-field instruction you are given. It is more specific than these rules \
and takes precedence over them."""

_TOOLS_MERGE = """\
You consolidate research notes about ONE tool in a comparison article into a single \
authoritative entry.
For each field you receive numbered claims taken from different source pages about the \
same tool.
Your job:
1. Drop any claim that is plainly about a different tool. Comparison pages leak; the \
entry must describe the named subject only.
2. Merge claims that make the same point, keeping the clearest and most specific \
wording. Never repeat the same point twice in different words.
3. Preserve every distinct substantive point. Do not drop information just to be brief.
4. Keep figures exactly as claimed, with their units and period. Never average, convert \
or round them. Two different values for the same thing are a conflict, not a range.
5. If sources genuinely contradict each other on a fact, set conflict=true and explain \
the disagreement in conflict_note, naming the differing values. Use the best-supported \
value as the value. Differing levels of detail are NOT a conflict; only incompatible \
facts are.
6. Introduce nothing that is not present in the claims. You have no other source.
7. NEVER use the tool name as the value for a field. The tool name is just the subject \
— it is never a valid answer to a field's question. If no substantive claims exist for \
a field, return an empty value.
8. Write in neutral, factual English with no marketing language.
9. Follow any per-field instruction you are given. It is more specific than these rules \
and takes precedence over them."""

_TOOLS_WEB = """\
You are a fact-checker settling disagreements about ONE tool in a comparison article.
You will be given the tool, a field, and the conflicting values that different pages \
claimed for it.
Search the web to establish which claim is correct today. The vendor's own pricing \
page, docs and changelog outrank comparison posts, listicles and affiliate reviews, \
which go stale fast and frequently mis-state competitors' pricing.
Rules:
1. Base the verdict only on what you actually found while searching. Never fall back on \
memory.
2. Confirm you are reading about the named tool, not the one it is being compared \
against.
3. Report figures exactly as the source states them, with their units and period, and \
say what they apply to. Never convert or round.
4. If the search confirms one of the claims, or gives a more accurate current value, \
set resolved=true and give that value.
5. If sources genuinely still disagree, or you cannot find authoritative confirmation, \
set resolved=false and leave the value empty. An honest 'unresolved' is far more useful \
than a confident guess.
6. A value that changed over time is not a contradiction to average out. Report what is \
true now and say so in reasoning.
7. In reasoning, state in 1-3 sentences what you found and which source settled it. Be \
specific about figures.
8. Write plain factual English. No marketing language.
9. Follow any per-field instruction you are given. It is more specific than these rules \
and takes precedence over them."""


DEFAULTS: dict[Scenario, PromptSet] = {
    Scenario.GENERAL: PromptSet(
        extract=_GENERAL_EXTRACT, merge=_GENERAL_MERGE, web=_GENERAL_WEB
    ),
    Scenario.TOOLS: PromptSet(
        extract=_TOOLS_EXTRACT, merge=_TOOLS_MERGE, web=_TOOLS_WEB
    ),
}

STAGES = ("extract", "merge", "web")


class PromptLibrary(BaseModel):
    """Saved overrides. A blank field means "use the built-in default"."""

    scenarios: dict[str, PromptSet] = Field(default_factory=dict)


_library: PromptLibrary | None = None


def _coerce_scenario(scenario: str | Scenario) -> Scenario:
    raw = scenario.value if isinstance(scenario, Enum) else scenario
    try:
        return Scenario(str(raw or "").strip().lower())
    except ValueError:
        return Scenario.GENERAL


def load_library() -> PromptLibrary:
    global _library
    if _library is None:
        try:
            raw = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
            _library = PromptLibrary.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValueError):
            _library = PromptLibrary()
    return _library


def default_prompts(scenario: str | Scenario) -> PromptSet:
    return DEFAULTS[_coerce_scenario(scenario)].model_copy()


def get_prompts(scenario: str | Scenario) -> PromptSet:
    """Saved override where one exists, built-in default otherwise, per stage."""
    sc = _coerce_scenario(scenario)
    base = DEFAULTS[sc]
    saved = load_library().scenarios.get(sc.value)
    if saved is None:
        return base.model_copy()
    return PromptSet(
        **{s: (getattr(saved, s) or "").strip() or getattr(base, s) for s in STAGES}
    )


def save_prompts(scenario: str | Scenario, prompts: PromptSet) -> None:
    """Persist an override. Text identical to the default is stored as blank so future
    default improvements are picked up instead of being frozen in."""
    sc = _coerce_scenario(scenario)
    base = DEFAULTS[sc]
    lib = load_library()
    lib.scenarios[sc.value] = PromptSet(
        **{
            s: (
                ""
                if (getattr(prompts, s) or "").strip() == getattr(base, s).strip()
                else (getattr(prompts, s) or "").strip()
            )
            for s in STAGES
        }
    )
    PROMPTS_FILE.write_text(
        json.dumps(lib.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def reset_prompts(scenario: str | Scenario) -> None:
    sc = _coerce_scenario(scenario)
    lib = load_library()
    lib.scenarios.pop(sc.value, None)
    PROMPTS_FILE.write_text(
        json.dumps(lib.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def is_customised(scenario: str | Scenario, stage: str) -> bool:
    sc = _coerce_scenario(scenario)
    saved = load_library().scenarios.get(sc.value)
    return bool(saved and (getattr(saved, stage, "") or "").strip())


def resolve(stage: str, scenario: str | Scenario, override: str = "") -> str:
    """The prompt actually sent to the model.

    `override` carries an unsaved edit from the UI so a prompt can be trialled without
    committing it to disk.
    """
    text = (override or "").strip()
    return text or getattr(get_prompts(scenario), stage)
