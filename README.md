# SEO Content Pipeline

Turns a list of reference URLs into a structured comparison dataset — one row per
product — with every claim traceable back to the page it came from.

```
URLs + nodes ──► scrape ──► extract (per source) ──► reconcile ──► review ──► export
                 crawl4ai    claims + quotes         merge, flag     QA        xlsx/csv
                                                     conflicts
```

## Setup

Requires Python 3.10+ (the code uses PEP 604 unions). A `.venv` is already created here
with Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m playwright install chromium   # ~115 MB, one time
```

Then set a key — either copy `.env.example` to `.env` and fill it in, or paste the key
into the sidebar at runtime.

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## Model providers

| Provider | JSON guarantee | Notes |
|---|---|---|
| `openai` | strict `json_schema` — enforced by the API | Most reliable. `gpt-4o-mini` is the default. |
| `anthropic` | forced tool call | Equally reliable; schema is the tool's input schema. |
| `deepseek` | loose `json_object` + local validation | Cheap and capable. Use `deepseek-chat`. |
| `ollama` | loose `json_object` + local validation | Local and free. Weakest on a 17-field schema. |

Each provider remembers its own key, endpoint and model, so switching between them never
sends one provider's credential to another's endpoint.

**DeepSeek specifics.** The API is OpenAI-compatible (`https://api.deepseek.com/v1`) but
has no strict-schema mode, so the JSON Schema is embedded in the system prompt and
validated locally. When a reply fails validation, the error is fed back and the model is
asked to fix that specific problem, rather than blindly resending the same prompt — which
otherwise tends to reproduce the same malformed output. Two consequences worth knowing:

- Embedding the schema costs roughly 1.4k extra prompt tokens per call on this 17-field
  brief (measured: 5.7k chars).
- `deepseek-reasoner` spends output tokens on hidden reasoning and ignores sampling
  parameters (the code omits them for it). It is slower and no better at filling a wide
  schema, so `deepseek-chat` is the right default for extraction.

DeepSeek is not automatically the cheapest option here. Extraction is input-heavy, and at
roughly $0.28/$0.42 per 1M tokens versus `gpt-4o-mini` at $0.15/$0.60, DeepSeek costs
slightly *more* on this workload. Pick it for quality or data-residency reasons rather than
on the assumption that it is cheaper, and verify current prices — they change often and the
built-in table is only indicative.

## Core idea: the schema is data, not code

The shape of the deliverable lives in `schemas/*.yaml`, not in Python. A content brief
becomes a schema file; a new brief or client never requires a code change. Each field
declares the question that defines it, the shape of the answer, and which input nodes are
allowed to answer it.

```yaml
- key: pricing_starting_from
  label: Pricing (Starting from)          # exact spreadsheet header
  question: What is the minimum price one can pay to use it meaningfully?
  guidance: Give the lowest recurring paid price with currency and period.
  shape: short_text                       # short_text | list | prose
  nodes: [Pricing]                        # only Pricing URLs may answer this
```

- `nodes: []` (omitted) means **every** node may answer the field.
- `fill_from: entity` copies the product name from the input sheet instead of spending an
  LLM call — used for the "Alternative name" column.
- `shape` drives both the JSON schema sent to the model and how the cell is rendered.

Edit schemas in the **Schema** tab (grid or raw YAML) and save to disk. The tab lints for
duplicate keys, fields pointing at nonexistent nodes, and nodes that feed nothing.

## Input format

One row per URL. Column names are matched loosely and the URL column is detected by
content, so pasting from a spreadsheet works.

| Product | Node | URL |
|---|---|---|
| n8n | Strengths, limitations, best for | https://… |
| n8n | Pricing | https://n8n.io/pricing/ |
| Zapier | Pricing | https://zapier.com/pricing |

`Product` is optional — supply a default in the UI if your sheet omits it. Node values
must match the schema's nodes (matching is case- and punctuation-insensitive, and close
typos are corrected with a warning). Duplicate rows and URLs differing only by `#fragment`
are collapsed automatically. See `sample_input.csv`.

## Why it is built this way

**JS rendering is mandatory.** A plain HTTP GET of `n8n.io/pricing` returns navigation
chrome and zero prices. With headless Chromium plus a settle delay we get the whole table
(`$20` Starter, `$50` Pro, `$800` Business). Pages that block the browser — one of the
nine sample URLs times out on an antibot check — fall back to static extraction
automatically. All nine succeed.

**Subject isolation.** Most reference pages for "alternatives" content are "A vs B"
comparisons, so the dominant accuracy risk is attributing the competitor's traits to your
product. The extraction prompt names the subject repeatedly and forbids describing
anything else.

**Provenance over convenience.** Extraction runs per source page, not per product, so
every claim keeps its URL. Each claim must carry a verbatim supporting quote, which is
then checked against the page text; an unverifiable quote is flagged in the UI and in the
Claims sheet as a possible invention.

**Conflicts surface instead of blending.** Two sources giving two different starting
prices is a fact about your research, not something to hide by concatenating both into one
cell. Such fields are marked `[CONFLICT]` and listed in Review for a human decision.

**Dedupe never merges different numbers.** Character-similarity dedupe rates "supports 400
integrations" and "supports 500 integrations" as 94% identical and silently destroys one.
Merging is therefore vetoed whenever two candidates contain different numbers.

**Caching, because you will re-run this a lot.** Fetched pages and LLM answers are cached
on disk, keyed on everything that could change the result (URL, model, temperature, and
the prompt text itself). Tuning a question re-runs only the affected fields. Re-running an
unchanged pipeline costs nothing.

**Nothing is lost on refresh.** Every stage writes the full run to `runs/*.json`. Reopen
any past run from the **Saved runs** tab.

## Running

The **Run** tab shows a preflight before spending anything: how many calls, a rough cost,
and — importantly — which schema fields no URL can answer. Those cells will be blank, and
it is better to know first.

Stage options let you re-extract without re-fetching, or re-merge without re-extracting.

## Output

The workbook has seven sheets:

| Sheet | Purpose |
|---|---|
| Dataset | One row per product. Paste straight into the client sheet. |
| Dataset (transposed) | Fields as rows — easier to read for wide schemas. |
| QA coverage | Fill rate, empty fields, conflicts, single-source fields per product. |
| Provenance | Every cell with its contributing URLs and merge method. |
| Claims | Every individual claim with its quote and verification status. |
| Pages | Fetch log: method used, character counts, failures. |
| Brief | The schema itself, as documentation. |

Review single-source and conflicting fields first — that is where errors concentrate.

## Layout

```
app.py                  Streamlit UI (6 tabs)
jobs.py                 background job + pollable progress bus
config.py               settings, resolved from .env then sidebar
schemas/*.yaml          output schemas (the content briefs)
pipeline/
  schema.py             FieldSpec / SchemaSpec, node resolution, linting
  models.py             pydantic domain models
  scraper.py            crawl4ai + static fallback, concurrent
  providers.py          OpenAI / Anthropic / Ollama, schema-validated JSON
  extractor.py          per-source claim extraction with quote verification
  reconciler.py         cross-source merge, conflict detection
  exporter.py           dataframes and the workbook
  runner.py             input parsing, preflight, orchestration
  cache.py / store.py   disk cache, run persistence
runs/ output/ .cache/   artifacts
```

## Notes and limits

- DeepSeek and Ollama have no strict JSON schema mode, so the schema goes in the prompt and
  is validated locally with a repair retry. Expect more retries than with OpenAI or
  Anthropic, and weaker results from small local models on a 17-field schema.
- Cost estimates are order-of-magnitude only, from a small hardcoded price table that will
  drift out of date. They ignore prompt-caching discounts, which DeepSeek in particular
  applies aggressively.
- Scraping is polite (one request at a time per domain, 1s delay, real user agent) but does
  not consult `robots.txt`. Check that the sites you target permit this.
- Page content genuinely varies between fetches on some sites (A/B tests, geo pricing). The
  cache makes a given run reproducible; clearing it may change results.

## Second stage (not built)

Sanity/CMS publishing. The reconciled `RunState` in `runs/*.json` is the natural handoff
point: it holds final values plus full provenance, so a publisher can map fields to a
Sanity schema and push documents without re-running extraction.
