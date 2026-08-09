"""SEO content pipeline - Streamlit front end.

Scrape reference pages, extract structured claims per source, reconcile them into one
row per product, review provenance, export.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Playwright browser bootstrap ──────────────────────────────────
# Chromium is needed for JS-rendered pages (pricing tables, SPAs).
# On Streamlit Cloud there is no Docker build step, so the user must
# trigger the install manually from the sidebar.  Once installed it
# persists across reruns (session state) but not across restarts.

_PW_PATH = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
if _PW_PATH:
    try:
        os.makedirs(_PW_PATH, exist_ok=True)
    except OSError:
        _PW_PATH = ""
if not _PW_PATH:
    _PW_PATH = os.path.join(os.environ.get("HOME", "/tmp"), ".cache", "ms-playwright")
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = _PW_PATH


def _chromium_is_installed() -> bool:
    """Check whether Chromium is already on disk.  Fast, no subprocess."""
    return bool(list(Path(_PW_PATH).glob("chromium*/*/chrome*")))


def _install_chromium() -> tuple[bool, str]:
    """Download and install Chromium.  Returns (success, error_message)."""
    from playwright.sync_api import sync_playwright

    if _chromium_is_installed():
        return True, ""

    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            return False, detail or f"exit code {result.returncode}"

        with sync_playwright() as p:
            p.chromium.launch(headless=True).close()
        return True, ""
    except Exception as e:
        return False, str(e)


from config import RUNS_DIR, SCHEMA_DIR, Config
from jobs import start_job
from pipeline import cache as cache_mod
from pipeline import exporter
from pipeline.brief import list_sheets, parse_brief_sheet
from pipeline.models import CUSTOM_INPUT_SCHEME, InputRow, RunState
from pipeline.prompts import (
    SCENARIO_LABELS,
    STAGES,
    PromptSet,
    default_prompts,
    get_prompts,
    is_customised,
    reset_prompts,
    save_prompts,
)
from pipeline.prompts import STAGE_LABELS as PROMPT_STAGE_LABELS
from pipeline.providers import (
    PROVIDERS,
    SEARCH_PROVIDERS,
    SUGGESTED_MODELS,
    build_provider,
    estimate_cost,
    estimate_search_cost,
)
from pipeline.runner import parse_input, preflight, run_pipeline
from pipeline.schema import (
    FieldShape,
    FieldSpec,
    FillFrom,
    ListOutput,
    Scenario,
    SchemaSpec,
    list_schemas,
    load_schema_by_name,
)
from pipeline.store import (
    BACKENDS,
    delete_run,
    get_db,
    get_settings,
    list_runs,
    load_all_run_files,
    load_run,
    runs_signature,
    save_run_file,
    set_settings,
)

st.set_page_config(
    page_title="SEO Content Pipeline | Mike - RINGHEL  ", page_icon="📑", layout="wide"
)

STRETCH = "stretch"


# ----------------------------------------------------------------- session bootstrap


def init_state() -> None:
    defaults = {
        "spec": None,
        "spec_source": "",
        "rows": [],
        "input_warnings": [],
        "run": None,
        "usages": {},
        "job": None,
        "job_stages": (),
        "preflight": None,
        "loaded_run_msg": "",
        "active_run_id": "",
        "chromium_installed": _chromium_is_installed(),
        "prompt_scenario": Scenario.GENERAL.value,
        # Unsaved prompt edits, keyed "<scenario>:<stage>". Kept separate from the saved
        # library so a prompt can be trialled on a run without being committed.
        "prompt_drafts": {},
        # Streamlit restores a text_area from its key and ignores `value`, so a reset
        # cannot repopulate the box in place. Bumping this changes the key, which builds
        # a genuinely new widget that does take the default.
        "prompt_nonce": 0,
        # Same problem, worse consequence: after loading or importing a schema the
        # editor would keep showing the previous one's name, nodes and fields, and
        # Apply would write those back over the new spec.
        "spec_nonce": 0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    if st.session_state.spec is None:
        available = list_schemas()
        if available:
            st.session_state.spec = SchemaSpec.load(available[0])
            st.session_state.spec_source = available[0].stem


def current_spec() -> SchemaSpec | None:
    return st.session_state.spec


def build_cfg() -> Config:
    """Assemble a Config from sidebar widgets."""
    cfg = Config()
    ss = st.session_state
    cfg.provider = ss.get("provider", cfg.provider)
    cfg.model = ss.get("model", cfg.model)
    cfg.temperature = ss.get("temperature", cfg.temperature)
    cfg.max_output_tokens = ss.get("max_output_tokens", cfg.max_output_tokens)
    cfg.llm_concurrency = ss.get("llm_concurrency", cfg.llm_concurrency)
    cfg.scrape_concurrency = ss.get("scrape_concurrency", cfg.scrape_concurrency)
    cfg.max_scrape_chars = ss.get("max_scrape_chars", cfg.max_scrape_chars)
    cfg.page_timeout_ms = int(ss.get("page_timeout_s", 60) * 1000)
    cfg.settle_delay_s = ss.get("settle_delay_s", cfg.settle_delay_s)
    cfg.use_cache = ss.get("use_cache", cfg.use_cache)
    cfg.reconcile_batch_size = ss.get("reconcile_batch_size", cfg.reconcile_batch_size)
    cfg.web_check_conflicts = ss.get("web_check_conflicts", cfg.web_check_conflicts)
    cfg.web_check_max_searches = ss.get(
        "web_check_max_searches", cfg.web_check_max_searches
    )

    # The schema decides the scenario; unsaved Advanced-tab edits for that scenario ride
    # along so a trial prompt is used without being saved to disk.
    spec = ss.get("spec")
    cfg.scenario = spec.scenario.value if spec else cfg.scenario
    drafts = ss.get("prompt_drafts") or {}
    cfg.extract_prompt_override = drafts.get(f"{cfg.scenario}:extract", "")
    cfg.merge_prompt_override = drafts.get(f"{cfg.scenario}:merge", "")
    cfg.web_prompt_override = drafts.get(f"{cfg.scenario}:web", "")

    # Credentials and endpoints are stored per provider, so switching provider cannot
    # leak the previous one's base URL or key into the new one.
    p = cfg.provider
    key = ss.get(f"api_key_{p}", "")
    base = ss.get(f"base_url_{p}", "")
    if p == "openai":
        cfg.openai_api_key = key or cfg.openai_api_key
        cfg.openai_base_url = base or cfg.openai_base_url
    elif p == "anthropic":
        cfg.anthropic_api_key = key or cfg.anthropic_api_key
    elif p == "deepseek":
        cfg.deepseek_api_key = key or cfg.deepseek_api_key
        cfg.deepseek_base_url = base or cfg.deepseek_base_url
    elif p == "ollama":
        cfg.ollama_base_url = base or cfg.ollama_base_url
    return cfg


def make_provider(cfg: Config):
    return build_provider(cfg.provider, cfg.key_for(), cfg.base_url_for(), cfg.model)


# -------------------------------------------------------------------------- sidebar


def render_sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        # ── Chromium browser ──────────────────────────────────────
        st.subheader("Browser")
        if ss.get("chromium_installed"):
            st.success("Chromium ready", icon="✅")
        else:
            st.warning("Chromium not installed", icon="⚠️")
            st.caption(
                "Needed for pages that build content with JavaScript "
                "(pricing tables, SPAs). Without it, a static fallback "
                "is used which may miss JS-rendered content."
            )
            if st.button("Install Chromium (~130 MB)", type="primary", width="stretch"):
                with st.spinner("Downloading Chromium…"):
                    ok, err = _install_chromium()
                if ok:
                    ss["chromium_installed"] = True
                    st.rerun()
                else:
                    st.error(f"Install failed: {err}")

        st.divider()

        # ── Model ─────────────────────────────────────────────────
        st.subheader("Model")
        provider = st.selectbox(
            "Provider",
            PROVIDERS,
            key="provider",
            help="OpenAI and Anthropic guarantee schema conformance in the API itself, so "
            "they are the most reliable. DeepSeek and Ollama only offer loose JSON "
            "mode, so the schema is validated locally and repaired on retry.",
        )
        default_cfg = Config()
        models = SUGGESTED_MODELS.get(provider, [])
        # Per-provider widget keys: a stale OpenAI model name must not survive a switch
        # to DeepSeek, and each provider should remember its own key and endpoint.
        st.selectbox(
            f"Model ({provider})",
            models,
            key=f"model_{provider}",
            accept_new_options=True,
            help="Type a name to use a model not listed here.",
        )
        ss["model"] = ss.get(f"model_{provider}") or (
            models[0] if models else default_cfg.model
        )

        env_keys = {
            "openai": default_cfg.openai_api_key,
            "anthropic": default_cfg.anthropic_api_key,
            "deepseek": default_cfg.deepseek_api_key,
        }

        # Streamlit discards state for widgets it did not render this run, so a typed key
        # would vanish when the user switches provider and back. Mirror entered values
        # into plain session slots and seed the widgets from them.
        def sticky(label: str, kind: str, fallback: str, **kw) -> str:
            wkey, skey = f"{kind}_{provider}", f"_saved_{kind}_{provider}"
            st.text_input(label, value=ss.get(skey, fallback), key=wkey, **kw)
            if ss.get(wkey):
                ss[skey] = ss[wkey]
            return ss.get(wkey, "")

        if provider in env_keys:
            sticky(
                "API key",
                "api_key",
                "",
                type="password",
                placeholder="set in .env, or paste here",
            )
            if env_keys[provider] and not ss.get(f"api_key_{provider}"):
                st.caption("Using key from .env")

        default_bases = {
            "openai": default_cfg.openai_base_url,
            "deepseek": default_cfg.deepseek_base_url,
            "ollama": default_cfg.ollama_base_url,
        }
        if provider in default_bases:
            sticky("Base URL", "base_url", default_bases[provider])

        if provider == "deepseek":
            st.caption(
                "DeepSeek has no strict-schema mode, so the schema is sent in the "
                "prompt and validated locally with an automatic repair retry. "
                "deepseek-v4-flash is the sensible default for extraction."
            )
        elif provider == "openai":
            st.caption(
                "gpt-5.x models reason before answering, so part of the output budget "
                "goes to hidden thinking and temperature is ignored. Reasoning effort is "
                "pinned low for extraction; raise 'Max output tokens' if calls truncate."
            )
        elif provider == "anthropic":
            st.caption(
                "Claude 5 models think by default and reject a custom temperature, so "
                "spend is steered with a low effort level instead. Part of the output "
                "budget goes to thinking, so keep 'Max output tokens' generous."
            )
        elif provider == "ollama":
            st.caption("Local model. Expect weaker results on a 17-field schema.")

        if st.button("Test connection", width=STRETCH, key="sb_test"):
            try:
                with st.spinner("Calling the model..."):
                    msg = make_provider(build_cfg()).smoke_test()
                st.success(msg)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

        # ── Web conflict resolution ───────────────────────────────
        st.divider()
        can_search = provider in SEARCH_PROVIDERS
        st.toggle(
            "Resolve conflicts with web search",
            key="web_check_conflicts",
            value=False,
            disabled=not can_search,
            help="After merging, take every field where sources disagree to the web. "
            "The model searches live pages — the vendor's own site first — and either "
            "settles the conflict with citations or reports that it could not.",
        )
        if not can_search:
            st.caption(
                f"Not available for {provider}: only OpenAI and Anthropic host a web "
                "search tool. Switch provider to enable this."
            )
        elif ss.get("web_check_conflicts"):
            st.number_input(
                "Max searches per conflict",
                1,
                10,
                4,
                1,
                key="web_check_max_searches",
                help="Hard cap per field. Each search is billed on top of tokens "
                "(about $0.01).",
            )
            st.caption(
                "Runs only on conflicting fields, so cost scales with disagreement, "
                "not schema size. Verdicts are cached and shown in Review."
            )

        with st.expander("Extraction settings"):
            st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, key="temperature")
            st.number_input(
                "Max output tokens",
                1000,
                32000,
                8000,
                1000,
                key="max_output_tokens",
                help="Too low truncates the JSON and the call fails.",
            )
            st.number_input("LLM concurrency", 1, 16, 4, 1, key="llm_concurrency")
            st.number_input(
                "Merge batch size",
                1,
                20,
                6,
                1,
                key="reconcile_batch_size",
                help="Fields merged per call. Larger is cheaper but risks truncation.",
            )

        with st.expander("Scraping settings"):
            st.number_input(
                "Browser concurrency", 1, 12, 4, 1, key="scrape_concurrency"
            )
            st.number_input(
                "Max chars per page", 5000, 200000, 40000, 5000, key="max_scrape_chars"
            )
            st.number_input("Page timeout (s)", 10, 180, 60, 10, key="page_timeout_s")
            st.number_input(
                "Settle delay (s)",
                0.0,
                10.0,
                2.5,
                0.5,
                key="settle_delay_s",
                help="Extra wait after load. Client-rendered pricing tables "
                "need this.",
            )

        with st.expander("Cache"):
            st.checkbox(
                "Use cache",
                value=True,
                key="use_cache",
                help="Reuses fetched pages and previous LLM answers. Keys include "
                "the model and prompts, so edits correctly miss the cache.",
            )
            n, nbytes = cache_mod.total_size()
            st.caption(f"{n} entries · {nbytes / 1e6:.1f} MB")
            if st.button("Clear cache", width=STRETCH, key="sb_clear_cache"):
                cleared = cache_mod.clear_all()
                st.success(f"Cleared: {cleared}")

        spec = current_spec()
        st.divider()
        if spec:
            st.caption(f"Schema **{spec.name}** · {len(spec.fields)} fields")
        else:
            st.warning("No schema loaded.")


# --------------------------------------------------------------------- schema editor


def fields_to_df(spec: SchemaSpec) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "section": f.section,
                "key": f.key,
                "label": f.label,
                "question": f.question,
                "shape": f.shape.value,
                "max_items": f.max_items,
                "guidance": f.guidance,
                "anchors": f.anchors,
                "custom_input": f.custom_input,
                "source_urls": "\n".join(f.source_urls),
                "source": f.fill_from.value,
            }
            for f in spec.fields
        ]
    )


def df_to_fields(
    df: pd.DataFrame, previous: SchemaSpec | None = None
) -> tuple[list[FieldSpec], list[str]]:
    # Legacy node assignments are not editable in the grid, so they are carried over
    # rather than silently dropped from a schema that still relies on them.
    prior_nodes = {f.key: f.nodes for f in (previous.fields if previous else [])}
    fields, errors = [], []
    for i, r in df.iterrows():
        key = str(r.get("key", "") or "").strip()
        if not key:
            continue
        try:
            fields.append(
                FieldSpec(
                    key=key,
                    label=str(r.get("label") or key).strip(),
                    question=str(r.get("question") or "").strip(),
                    shape=FieldShape(str(r.get("shape") or "prose").strip()),
                    nodes=prior_nodes.get(key, []),
                    max_items=int(r.get("max_items") or 10),
                    guidance=str(r.get("guidance") or "").strip(),
                    fill_from=FillFrom(str(r.get("source") or "extract").strip()),
                    section=str(r.get("section") or "").strip(),
                    anchors=str(r.get("anchors") or "").strip(),
                    custom_input=str(r.get("custom_input") or "").strip(),
                    source_urls=[
                        u.strip()
                        for u in str(r.get("source_urls") or "").splitlines()
                        if u.strip()
                    ],
                )
            )
        except Exception as e:
            errors.append(f"Row {i + 1} ({key}): {e}")
    return fields, errors


def _chips(labels: list[str]) -> str:
    return (
        "".join(
            f'<span style="display:inline-block;background:#2d3748;color:#e2e8f0;'
            f"border-radius:4px;padding:1px 8px;margin:2px 4px 2px 0;"
            f'font-size:0.85em;white-space:nowrap;">{lbl}</span>'
            for lbl in labels
        )
        or '<span style="color:#718096;font-style:italic;">(none)</span>'
    )


def render_routing_view(spec: SchemaSpec) -> None:
    """Show which questions each source is asked."""
    by_key = {f.key: f for f in spec.fields}

    if spec.has_url_map:
        mapped = spec.url_map()
        with st.expander(f"Source → field routing ({len(mapped)} URLs)"):
            st.caption(
                "One row per distinct URL. A URL listed under several fields is still "
                "fetched once and extracted once, answering only these questions."
            )
            for url, keys in mapped.items():
                labels = [by_key[k].label for k in keys if k in by_key]
                st.markdown(
                    f'<div style="margin:6px 0;">'
                    f'<div style="font-weight:600;color:#e2e8f0;font-size:0.85em;'
                    f'word-break:break-all;">{url}</div>{_chips(labels)}</div>',
                    unsafe_allow_html=True,
                )

            custom = spec.custom_input_fields()
            if custom:
                st.markdown("**Analyst-supplied evidence (no fetch)**")
                st.markdown(_chips([f.label for f in custom]), unsafe_allow_html=True)

        unmapped = [
            f.label
            for f in spec.fields
            if f.fill_from == FillFrom.EXTRACT
            and not f.source_urls
            and not f.custom_input.strip()
        ]
        if unmapped:
            st.warning(
                f"{len(unmapped)} field(s) have no source and will stay empty: "
                + ", ".join(unmapped[:8])
                + ("…" if len(unmapped) > 8 else "")
            )
        return

    with st.expander("Node → field routing (legacy)"):
        for node in spec.all_nodes():
            st.markdown(
                f'<div style="margin:4px 0;">'
                f'<span style="font-weight:600;color:#e2e8f0;margin-right:8px;">'
                f"{node}</span>"
                f"{_chips([f.label for f in spec.fields_for_node(node)])}</div>",
                unsafe_allow_html=True,
            )


def render_brief_import() -> None:
    """Turn a 'data structure' workbook into a schema, one sheet at a time.

    The workbook holds one sheet per scenario, and they need different prompts, so they
    are imported as separate schemas rather than merged into one.
    """
    with st.expander("Import a brief workbook (.xlsx)"):
        st.caption(
            "Expects the client's data-structure columns: Article - H2, Article - H3, "
            "Key, Prompt, Anchors, Source urls, Custom input. Each sheet becomes one "
            "schema — the General sheet and the Tools sheet use different system "
            "prompts, so import them separately."
        )
        up = st.file_uploader("Workbook", type=["xlsx", "xlsm"], key="brief_uploader")
        if not up:
            return

        data = up.getvalue()
        try:
            sheets = list_sheets(data)
        except Exception as e:
            st.error(f"Could not read workbook: {e}")
            return

        c1, c2, c3 = st.columns([2, 1, 1])
        sheet = c1.selectbox("Sheet", sheets, key="brief_sheet")
        name = c2.text_input("Schema name", value="", key="brief_name")
        entity = c3.text_input("Entity label", value="Product", key="brief_entity")

        if st.button("Import sheet", type="primary", width=STRETCH, key="brief_import"):
            try:
                spec, warns = parse_brief_sheet(
                    data, sheet, name.strip(), entity.strip() or "Product"
                )
            except Exception as e:
                st.error(f"Import failed: {e}")
                return
            st.session_state.spec = spec
            st.session_state.spec_source = spec.name
            st.session_state.spec_nonce += 1
            for w in warns:
                st.warning(w)
            st.success(
                f"Imported {len(spec.fields)} field(s) from {sheet!r} as the "
                f"{spec.scenario.value} scenario. Review below, then Save to disk."
            )
            st.rerun()


def tab_schema() -> None:
    st.subheader("Output schema")
    st.caption(
        "The deliverable's shape lives here as data, not in code. Each row is one output "
        "column: the question that defines it, the shape of the answer, and the source "
        "URLs or pasted evidence it is answered from."
    )

    files = list_schemas()
    names = [p.stem for p in files]
    col1, col2 = st.columns([2, 1])
    with col1:
        picked = st.selectbox(
            "Saved schemas",
            names or ["(none)"],
            index=(
                names.index(st.session_state.spec_source)
                if st.session_state.spec_source in names
                else 0
            ),
        )
    with col2:
        st.write("")
        if st.button("Load", width=STRETCH, disabled=not names, key="schema_load"):
            st.session_state.spec = SchemaSpec.load(SCHEMA_DIR / f"{picked}.yaml")
            st.session_state.spec_source = picked
            st.session_state.spec_nonce += 1
            st.success(f"Loaded {picked}")
            st.rerun()

    render_brief_import()

    spec = current_spec()
    if spec is None:
        st.info("No schema available. Create one below.")
        return

    # Widget keys carry a nonce that changes whenever the spec is replaced, so loading
    # or importing a schema rebuilds these inputs instead of showing the previous one.
    n = st.session_state.spec_nonce
    a, b, c, d = st.columns(4)
    a.text_input("Schema name", value=spec.name, key=f"spec_name_{n}")
    b.text_input(
        "Entity label",
        value=spec.entity_label,
        key=f"spec_entity_{n}",
        help="What one row represents. 'Product' here.",
    )
    scenarios = [s.value for s in Scenario]
    c.selectbox(
        "Scenario",
        scenarios,
        index=scenarios.index(spec.scenario.value),
        key=f"spec_scenario_{n}",
        format_func=lambda v: SCENARIO_LABELS[Scenario(v)],
        help="Which family of system prompts a run on this schema uses. Edit the "
        "prompts themselves in the Advanced tab.",
    )
    d.selectbox(
        "List rendering",
        [o.value for o in ListOutput],
        index=[o.value for o in ListOutput].index(spec.list_output.value),
        key=f"spec_listout_{n}",
        help="How list fields are written into a cell.",
    )
    if spec.has_url_map:
        st.caption(
            f"Routing: **{len(spec.url_map())} source URL(s)** mapped to field keys, "
            "many-to-many. Each URL is fetched once and asked only its own questions."
        )
    else:
        st.text_area(
            "Nodes (one per line)",
            value="\n".join(spec.nodes),
            key=f"spec_nodes_{n}",
            height=90,
            help="Legacy routing, used only while no field has source URLs. Fill in "
            "'source urls' below to route by key instead.",
        )

    st.markdown("**Fields**")
    edited = st.data_editor(
        fields_to_df(spec),
        num_rows="dynamic",
        width=STRETCH,
        key=f"field_editor_{n}",
        column_config={
            "section": st.column_config.TextColumn(
                "section (H2)",
                help="The article heading this field sits under. Presentational only — "
                "routing happens through source urls.",
                width="small",
            ),
            "key": st.column_config.TextColumn(
                "key",
                help="snake_case identifier. Changing it invalidates cached "
                "extractions for the field.",
                width="small",
            ),
            "label": st.column_config.TextColumn(
                "label (H3)", help="Exact spreadsheet header."
            ),
            "question": st.column_config.TextColumn(
                "question",
                help="Defines the field. This text is sent to the model.",
                width="large",
            ),
            "shape": st.column_config.SelectboxColumn(
                "shape", options=[s.value for s in FieldShape], width="small"
            ),
            "max_items": st.column_config.NumberColumn(
                "max", min_value=1, max_value=50, width="small"
            ),
            "guidance": st.column_config.TextColumn("guidance", width="large"),
            "anchors": st.column_config.TextColumn(
                "anchors",
                width="large",
                help="Angles the brief insists on covering. Sent to the model as "
                "'Must cover'.",
            ),
            "custom_input": st.column_config.TextColumn(
                "custom input",
                width="large",
                help="Evidence you pasted by hand (a forum reply, a note from a call). "
                "Becomes an extra source for this field, like one more URL — no "
                "scraping needed.",
            ),
            "source_urls": st.column_config.TextColumn(
                "source urls",
                width="large",
                help="URLs this field is answered from, one per line. The same URL may "
                "appear under several fields; it is still fetched only once.",
            ),
            "source": st.column_config.SelectboxColumn(
                "source",
                options=[f.value for f in FillFrom],
                width="small",
                help="'entity' copies the product name from the input sheet instead of "
                "spending a call.",
            ),
        },
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("Apply changes", type="primary", width=STRETCH, key="schema_apply"):
        fields, errors = df_to_fields(edited, spec)
        if errors:
            for e in errors:
                st.error(e)
        else:
            try:
                raw_nodes = st.session_state.get(f"spec_nodes_{n}")
                new_spec = SchemaSpec(
                    name=st.session_state[f"spec_name_{n}"].strip() or spec.name,
                    entity_label=(
                        st.session_state[f"spec_entity_{n}"].strip() or "Product"
                    ),
                    description=spec.description,
                    scenario=Scenario(st.session_state[f"spec_scenario_{n}"]),
                    nodes=(
                        [ln.strip() for ln in raw_nodes.splitlines() if ln.strip()]
                        if raw_nodes is not None
                        else spec.nodes
                    ),
                    list_output=ListOutput(st.session_state[f"spec_listout_{n}"]),
                    fields=fields,
                )
                st.session_state.spec = new_spec
                st.success("Schema updated for this session.")
            except Exception as e:
                st.error(f"Invalid schema: {e}")

    if c2.button("Save to disk", width=STRETCH, key="schema_save"):
        s = current_spec()
        path = s.save(SCHEMA_DIR / f"{s.name}.yaml")
        st.session_state.spec_source = s.name
        st.success(f"Saved {path.name}")

    problems = spec.lint()
    if problems:
        with c3:
            st.warning(f"{len(problems)} schema issue(s)")
        for p in problems:
            st.warning(p)
    else:
        c3.success("Schema looks consistent.")

    render_routing_view(spec)

    with st.expander("YAML"):
        yaml_text = spec.to_yaml_text()
        st.code(yaml_text, language="yaml", line_numbers=True)

        with st.popover("✏️ Edit YAML"):
            text = st.text_area(
                "Schema YAML", value=yaml_text, height=320, key="spec_yaml"
            )
            if st.button("Apply YAML", width=STRETCH, key="schema_apply_yaml"):
                try:
                    st.session_state.spec = SchemaSpec.from_yaml_text(text)
                    st.session_state.spec_nonce += 1
                    st.success("Applied.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid YAML: {e}")
        st.download_button(
            "Download YAML",
            data=yaml_text,
            file_name=f"{spec.name}.yaml",
            width=STRETCH,
            key="schema_dl_yaml",
        )


# ----------------------------------------------------------------------- input tab


def tab_input() -> None:
    st.subheader("Reference URLs")
    st.caption(
        "One row per URL: which product it describes and which fields it answers. "
        "Usually you just press *Seed input from schema* — the brief already lists the "
        "URLs. Columns are matched loosely and the URL column is detected by content."
    )

    spec = current_spec()
    src = st.radio(
        "Source",
        ["Upload file", "Paste table"],
        horizontal=True,
        label_visibility="collapsed",
        key="input_source",
    )

    default_product = st.text_input(
        "Default product name",
        key="input_default_product",
        help="Used for rows with no Product column. Required if your sheet lacks one.",
    )

    # The brief workbook already nominates URLs per field, so a run can start straight
    # from the schema without the user rebuilding that list by hand.
    seeded = spec.seed_rows(default_product.strip()) if spec else []
    if seeded:
        if st.button(
            f"Seed input from schema ({len(seeded)} URL(s))",
            type="primary",
            disabled=not default_product.strip(),
            key="input_seed",
            help="Uses the 'source urls' on each schema field. A URL listed under "
            "several fields becomes one row answering all of them.",
        ):
            st.session_state.rows = [
                InputRow(product=p, url=u, keys=k) for p, u, k in seeded
            ]
            st.session_state.input_warnings = []
            st.session_state.preflight = None
            st.rerun()
        if not default_product.strip():
            st.caption("Enter a product name above to seed from the schema.")

    data = None
    filename = ""
    if src == "Upload file":
        up = st.file_uploader(
            "CSV, TSV or Excel",
            type=["csv", "tsv", "txt", "xlsx", "xls"],
            key="input_uploader",
        )
        if up:
            data, filename = up.read(), up.name
    else:
        pasted = st.text_area(
            "Paste rows (tab or comma separated, with a header row)",
            height=200,
            key="input_paste",
            placeholder=(
                "Product\tKey\tURL\n" "n8n\tpricing_format\thttps://n8n.io/pricing/"
            ),
            help="A 'Key' column names the fields a URL answers; repeat the URL on "
            "several rows, or list keys separated by commas. Without it, the schema's "
            "own source urls decide.",
        )
        if pasted.strip():
            data, filename = pasted, "pasted.tsv"

    if st.button("Load input", type="primary", disabled=data is None, key="input_load"):
        try:
            rows, warns = parse_input(data, filename, default_product.strip())
            st.session_state.rows = rows
            st.session_state.input_warnings = warns
            st.session_state.preflight = None
            st.success(f"Loaded {len(rows)} rows.")
        except Exception as e:
            st.error(str(e))

    sample = Path("sample_input.csv")
    if sample.exists() and not st.session_state.rows:
        if st.button("Load bundled sample (n8n, 9 URLs)", key="input_sample"):
            rows, warns = parse_input(sample.read_bytes(), sample.name)
            st.session_state.rows = rows
            st.session_state.input_warnings = warns
            st.rerun()

    rows = st.session_state.rows
    if not rows:
        return

    for w in st.session_state.input_warnings:
        st.warning(w)

    label_of = {f.key: f.label for f in spec.fields} if spec else {}
    df = pd.DataFrame(
        [
            {
                "product": r.product,
                "url": r.url,
                "answers": ", ".join(label_of.get(k, k) for k in r.keys)
                or (f"node: {r.node}" if r.node else "(routed by schema)"),
            }
            for r in rows
        ]
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("URLs", len(rows))
    m2.metric("Products", df["product"].nunique())
    m3.metric("Fields covered", len({k for r in rows for k in r.keys}))
    st.dataframe(df, width=STRETCH, hide_index=True)

    if spec is None:
        st.warning("Load a schema to run preflight checks.")
        return

    custom = spec.custom_input_fields()
    if custom:
        st.caption(
            f"Plus {len(custom)} field(s) answered from custom input you pasted in the "
            "schema — added automatically at run time, no fetch needed."
        )

    st.markdown("**Preflight**")
    pf = preflight(rows, spec, build_cfg())
    st.session_state.preflight = pf
    if pf["problems"]:
        for p in pf["problems"]:
            st.warning(p)
    else:
        st.success("Every schema field has at least one source.")
    for n in pf["notes"]:
        st.info(n)


# ------------------------------------------------------------------------- run tab


def render_progress(job) -> None:
    for stage, sp in job.bus.snapshot():
        label = f"**{stage}** {sp.done}/{sp.total or '?'}"
        st.progress(sp.fraction, text=f"{label} — {sp.message[:90]}")


def tab_run() -> None:
    st.subheader("Run")
    spec = current_spec()
    rows = st.session_state.rows
    if spec is None or not rows:
        st.info("Load a schema and an input sheet first.")
        return

    cfg = build_cfg()
    pf = st.session_state.preflight or preflight(rows, spec, cfg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("URLs", pf["urls"])
    c2.metric("Products", len(pf["products"]))
    c3.metric("Extraction calls", pf["est_extract_calls"])
    c4.metric("Merge calls", f"≤{pf['est_reconcile_calls']}")

    est = estimate_cost(
        cfg.model, pf["est_prompt_tokens"], pf["est_extract_calls"] * 700
    )
    if est is not None:
        st.caption(
            f"Very rough first-run estimate for {cfg.model}: **${est:.2f}**. Cached "
            "pages and answers make re-runs far cheaper."
        )

    use_llm_merge = st.toggle(
        "Merge with the LLM",
        value=True,
        key="run_use_llm",
        help="On: the model consolidates claims from multiple sources and flags "
        "contradictions. Off: free mechanical dedupe, no semantic merging.",
    )

    if cfg.web_check_conflicts:
        if cfg.provider in SEARCH_PROVIDERS:
            st.caption(
                "Web conflict resolution is **on** — after merging, each conflicting "
                "field is checked against live web sources. Turn it off in the sidebar."
            )
        else:
            st.warning(
                f"Web conflict resolution is on but {cfg.provider} has no web search "
                "tool, so it will be skipped. Switch to OpenAI or Anthropic to use it."
            )

    opts = {
        "Everything (scrape → extract → merge)": ("scrape", "extract", "reconcile"),
        "Re-extract and merge (reuse fetched pages)": ("extract", "reconcile"),
        "Re-merge only (reuse extractions)": ("reconcile",),
        "Web-check conflicts only (reuse merged results)": (),
    }
    choice = st.radio("Stages", list(opts), horizontal=False, key="run_stages")
    stages = opts[choice]

    prior: RunState | None = st.session_state.run
    full = ("scrape", "extract", "reconcile")

    if not stages:
        # Web-check-only reuses the merged results as they stand, so it needs both a
        # completed run to read and the toggle that actually performs the check.
        if not cfg.web_check_conflicts:
            st.warning(
                "This option runs only the web check, which is switched off in the "
                "sidebar. Enable the toggle first."
            )
        if prior is None or not prior.results:
            st.warning(
                "No merged results in this session to check. Run the pipeline or load a "
                "completed run first."
            )
    elif stages != full and prior is None:
        st.warning(
            "No previous run in this session; the full pipeline will run instead."
        )
        stages = full

    job = st.session_state.job

    if job is None or job.done:
        if st.button("Start run", type="primary", width=STRETCH, key="run_start"):
            try:
                provider = make_provider(cfg)
            except Exception as e:
                st.error(f"Provider error: {e}")
                return
            st.session_state.job = start_job(
                run_pipeline,
                rows=rows,
                spec=spec,
                cfg=cfg,
                provider=provider,
                use_llm_merge=use_llm_merge,
                state=prior if stages != full else None,
                stages=stages,
            )
            st.session_state.job_stages = stages
            st.rerun()

    job = st.session_state.job
    if job is None:
        return

    if job.running or not job.done:
        st.info("Running. This page updates as stages complete.")
        render_progress(job)
        time.sleep(0.6)
        st.rerun()
        return

    # Finished
    if job.error:
        st.error(job.error)
        with st.expander("Traceback"):
            st.code(job.result.get("traceback", ""))
        if st.button("Dismiss", key="run_dismiss"):
            st.session_state.job = None
            st.rerun()
        return

    value = job.result.get("value")
    if value:
        state, usages = value
        st.session_state.run = state
        st.session_state.usages = usages
        # Keep the Saved runs tab pointing at whatever is actually in the session.
        st.session_state.active_run_id = state.run_id
    render_progress(job)
    st.success("Run complete. See the Review tab.")

    state = st.session_state.run
    if state:
        s = state.stats()
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Pages OK", s["pages_ok"])
        d2.metric("Pages failed", s["pages_failed"], delta_color="inverse")
        d3.metric("Extractions", s["extractions"])
        d4.metric("Extraction errors", s["extraction_errors"], delta_color="inverse")

        total_in = total_out = total_searches = 0
        for name, u in (st.session_state.usages or {}).items():
            total_in += u.prompt_tokens
            total_out += u.completion_tokens
            total_searches += u.searches
            line = (
                f"{name}: {u.calls} live call(s), {u.cached} cached, {u.errors} error(s), "
                f"{u.prompt_tokens:,} in / {u.completion_tokens:,} out tokens"
            )
            if u.searches:
                line += f", {u.searches} web search(es)"
            st.caption(line)
        actual = estimate_cost(state.model, total_in, total_out) or 0.0
        search_cost = estimate_search_cost(state.provider, total_searches) or 0.0
        if actual or search_cost:
            note = f"Approximate spend this run: **${actual + search_cost:.3f}**"
            if search_cost:
                note += f" (including ${search_cost:.3f} of web searches)"
            st.caption(note)

        if state.warnings:
            with st.expander(f"{len(state.warnings)} warning(s)"):
                for w in state.warnings:
                    st.write(f"- {w}")

    if st.button("Reset run state", key="run_reset"):
        st.session_state.job = None
        st.rerun()


# ---------------------------------------------------------------------- review tab


def render_web_verdict(rf, key_prefix: str) -> None:
    """Show what the web check concluded for one field, if it ran."""
    v = getattr(rf, "web", None)
    if v is None:
        return

    st.divider()
    if v.error:
        st.error(f"Web check failed: {v.error}")
        return

    if v.resolved:
        st.success("Settled by web search")
        if rf.value_before_web:
            st.caption(f"Before the check: {rf.value_before_web}")
    else:
        st.info(
            "The web check ran but could not settle this — the sources still disagree, "
            "so the merged value is unchanged and needs a human decision."
        )
    if v.reasoning:
        st.markdown(f"> {v.reasoning}")
    if v.citations:
        st.caption("Pages consulted:")
        for url in v.citations[:8]:
            st.markdown(f"- [{url[:90]}]({url})")
    else:
        st.caption(
            "The model returned no citations, so this verdict cannot be traced to a "
            "page. Treat it as unverified."
        )


def tab_review() -> None:
    st.subheader("Review")
    state: RunState | None = st.session_state.run
    spec = current_spec()
    if state is None or not state.results or spec is None:
        st.info("No results yet. Run the pipeline first.")
        return

    cov = exporter.build_coverage(state.results, spec)
    st.markdown("**Quality**")
    st.dataframe(cov, width=STRETCH, hide_index=True)

    conflicts = [
        (r.product, f.label, r.fields[f.key])
        for r in state.results
        for f in spec.fields
        if r.fields.get(f.key) and r.fields[f.key].conflict
    ]
    if conflicts:
        settled = sum(1 for _, _, rf in conflicts if rf.web and rf.web.resolved)
        msg = (
            f"{len(conflicts)} field(s) where sources disagree. These need a human "
            "decision before publishing."
        )
        if settled:
            msg += f" {settled} of them were settled by a web check."
        st.warning(msg)
        for product, label, rf in conflicts:
            mark = "✅ " if rf.web and rf.web.resolved else ""
            with st.expander(f"{mark}{product} · {label}"):
                st.write(rf.conflict_note or "(no note)")
                st.text_area(
                    "Merged value",
                    rf.value,
                    height=100,
                    key=f"cf_{product}_{label}",
                    disabled=True,
                )
                st.caption("Sources: " + ", ".join(rf.sources))
                render_web_verdict(rf, f"cf_{product}_{label}")

    st.markdown("**Dataset**")
    transposed = st.toggle(
        "Transpose (fields as rows)",
        value=True,
        key="review_transpose",
        help="Easier to read for wide schemas.",
    )
    ds = (
        exporter.build_dataset_transposed(state.results, spec)
        if transposed
        else exporter.build_dataset(state.results, spec)
    )
    st.dataframe(ds, width=STRETCH, hide_index=True, height=460)

    st.markdown("**Trace a field back to its sources**")
    c1, c2 = st.columns(2)
    product = c1.selectbox(
        "Product", [r.product for r in state.results], key="review_product"
    )
    label_to_key = {f.label: f.key for f in spec.fields}
    label = c2.selectbox("Field", list(label_to_key), key="review_field")
    key = label_to_key[label]

    result = next((r for r in state.results if r.product == product), None)
    rf = result.fields.get(key) if result else None
    if rf is None:
        st.info("Nothing recorded for this field.")
        return

    st.text_area(
        "Merged value",
        rf.value or "(empty)",
        height=130,
        disabled=True,
        key=f"trace_value_{product}_{key}",
    )
    a, b, c = st.columns(3)
    a.metric("Sources", rf.source_count)
    b.metric("Merge method", rf.method or "-")
    c.metric("Conflict", "yes" if rf.conflict else "no")

    render_web_verdict(rf, f"trace_{product}_{key}")

    # Explain *why* a value is empty so the user knows whether it's
    # a data gap (no sources covered this) or a processing failure.
    if rf.is_empty:
        if rf.method == "empty":
            st.info(
                "No source page covered this field. The cell is legitimately empty."
            )
        elif rf.method == "mechanical" and rf.conflict_note:
            st.warning(rf.conflict_note)
        else:
            st.info("No claims found for this field across any source.")

    st.caption("Individual claims behind this value:")
    any_claim = False
    for ext in state.extractions:
        if ext.product != product:
            continue
        claim = ext.claims.get(key)
        if not claim or not claim.found:
            continue
        any_claim = True
        src = (
            "analyst-supplied evidence"
            if ext.url.startswith(CUSTOM_INPUT_SCHEME)
            else ext.url[:95]
        )
        with st.expander(src):
            for v in claim.values:
                st.write(f"- {v}")
            if claim.quote:
                st.caption("Supporting quote:")
                st.markdown(f"> {claim.quote}")
            if not claim.quote_verified:
                st.error(
                    "This quote could not be located in the page text, so the claim may "
                    "be paraphrased or invented. Verify before publishing."
                )
    if not any_claim:
        st.info("No source claimed this field. The cell is legitimately empty.")


# ---------------------------------------------------------------------- export tab


def tab_export() -> None:
    st.subheader("Export")
    state: RunState | None = st.session_state.run
    spec = current_spec()
    if state is None or not state.results or spec is None:
        st.info("No results yet.")
        return

    ds = exporter.build_dataset(state.results, spec)
    stamp = exporter.timestamp()
    st.caption(
        "The workbook carries the dataset plus QA, provenance, per-claim quotes, fetch "
        "log and the brief, so every cell can be defended."
    )

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Dataset CSV",
        ds.to_csv(index=False).encode("utf-8"),
        file_name=f"{spec.name}_{stamp}.csv",
        mime="text/csv",
        width=STRETCH,
        key="exp_csv",
    )
    c2.download_button(
        "Full workbook (xlsx)",
        exporter.build_workbook(state.results, spec, state.extractions, state.pages),
        file_name=f"{spec.name}_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width=STRETCH,
        key="exp_xlsx",
    )
    c3.download_button(
        "Run JSON",
        state.model_dump_json(indent=2).encode("utf-8"),
        file_name=f"{state.run_id}.json",
        mime="application/json",
        width=STRETCH,
        key="exp_json",
    )

    st.dataframe(ds, width=STRETCH, hide_index=True)
    with st.expander("Provenance table"):
        st.dataframe(
            exporter.build_provenance(state.results, spec),
            width=STRETCH,
            hide_index=True,
        )
    with st.expander("Claims with quotes"):
        st.dataframe(
            exporter.build_claims(state.extractions, spec),
            width=STRETCH,
            hide_index=True,
        )
    with st.expander("Fetch log"):
        st.dataframe(exporter.build_pages(state.pages), width=STRETCH, hide_index=True)


# ------------------------------------------------------------------------ runs tab


STAGE_LABELS = {
    "created": "Created",
    "scraped": "Scraped",
    "extracted": "Extracted",
    "reconciled": "Complete",
    "unreadable": "Unreadable",
}


@st.cache_data(show_spinner=False)
def _runs_index(_signature: tuple) -> list[dict]:
    """Cached run index. `_signature` is the cache key, not data (leading underscore
    keeps Streamlit from hashing it as a value)."""
    return list_runs()


def _fmt_when(iso: str) -> str:
    """Render a stored ISO timestamp as a short local-ish label."""
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    if dt.tzinfo is not None:
        dt = dt.astimezone()
    today = datetime.now().astimezone().date()
    if dt.date() == today:
        return f"Today {dt:%H:%M}"
    if (today - dt.date()).days == 1:
        return f"Yesterday {dt:%H:%M}"
    return f"{dt:%d %b %H:%M}"


def apply_loaded_run(state: RunState, run_id: str) -> None:
    """Make a run from disk the active session state.

    A run is only reviewable alongside the schema it was produced with -- the review,
    export and re-run paths all read field keys from the spec -- so the run's own schema
    and input rows are restored too, not just the results.
    """
    notes: list[str] = []
    if state.schema_name and state.schema_name != (st.session_state.spec_source or ""):
        try:
            st.session_state.spec = load_schema_by_name(state.schema_name)
            st.session_state.spec_source = state.schema_name
            notes.append(f"schema `{state.schema_name}` loaded")
        except FileNotFoundError:
            notes.append(
                f"schema `{state.schema_name}` is no longer on disk, so the currently "
                "loaded schema is used and columns may not line up"
            )

    st.session_state.run = state
    st.session_state.rows = list(state.inputs)
    st.session_state.input_warnings = []
    st.session_state.usages = {}
    st.session_state.preflight = None
    st.session_state.job = None
    st.session_state.job_stages = ()
    st.session_state.active_run_id = run_id

    where = "Review" if state.results else "Run"
    st.session_state.loaded_run_msg = (
        f"Loaded **{run_id}** — {STAGE_LABELS.get(state.stage, state.stage)}, "
        f"{len(state.inputs)} URL(s)."
        + (" " + "; ".join(notes).capitalize() + "." if notes else "")
        + f" Continue in the **{where}** tab."
    )


def tab_runs() -> None:
    st.subheader("Saved runs")
    st.caption(
        "Every stage is written to disk, so a refresh never loses work. Loading a run "
        "also restores its schema and input sheet, so you can review it or re-run just "
        "the last stage."
    )

    # The client asked where runs live: there is no account, so say so plainly and name
    # the actual folder rather than leaving it to be guessed.
    s = get_settings()
    where = {
        "files": f"JSON files in `{RUNS_DIR}`",
        "db": f"the database `{s.resolved_url().split('://')[0]}`",
        "both": f"JSON files in `{RUNS_DIR}` **and** the database",
    }[s.backend]
    st.info(
        f"Runs are saved on the machine running this app — currently {where}. "
        "There is no login: anything listed here survives a refresh, a browser restart "
        "and a reboot, and is only lost if those files are deleted. Change the location "
        "in the **Storage** tab.",
        icon="💾",
    )

    runs = _runs_index(runs_signature())
    if not runs:
        st.warning(
            "No saved runs yet. A run appears here as soon as its first stage finishes, "
            "so if you expected one, the run either never started or failed during "
            "scraping — check the Run tab for an error."
        )
        return

    if st.session_state.get("loaded_run_msg"):
        st.success(st.session_state.pop("loaded_run_msg"))

    active = st.session_state.get("active_run_id") or ""
    df = pd.DataFrame(runs)
    # A plain glyph rather than a checkbox column: the grid already renders Streamlit's
    # own selection checkbox in the first column, and two checkboxes side by side read as
    # two competing controls.
    df.insert(0, "live", ["●" if r["run_id"] == active else "" for r in runs])
    df["stage_label"] = df["stage"].map(lambda s: STAGE_LABELS.get(s, s))
    df["when"] = df["updated_at"].where(df["updated_at"] != "", df["created_at"])
    df["when"] = df["when"].map(_fmt_when)
    df["fetched"] = [
        "—" if r["urls"] == 0 else f"{r['pages_ok']}/{r['urls']}" for r in runs
    ]

    st.caption(
        f"{len(runs)} run(s), newest first. Tick a row to inspect it; ● marks the run "
        "loaded in this session."
    )
    event = st.dataframe(
        df,
        width=STRETCH,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="runs_table",
        column_order=[
            "live",
            "run_id",
            "when",
            "stage_label",
            "source",
            "schema",
            "model",
            "products",
            "urls",
            "fetched",
            "results",
            "warnings",
            "size_kb",
        ],
        column_config={
            "live": st.column_config.TextColumn(
                "", width="small", help="The run currently loaded in this session."
            ),
            "run_id": st.column_config.TextColumn("Run", width="medium"),
            "source": st.column_config.TextColumn(
                "Where", width="small", help="Which backend this run was read from."
            ),
            "when": st.column_config.TextColumn("Last saved", width="small"),
            "stage_label": st.column_config.TextColumn(
                "Stage",
                width="small",
                help="How far the pipeline got. Only 'Complete' runs have a dataset.",
            ),
            "schema": st.column_config.TextColumn("Schema", width="small"),
            "model": st.column_config.TextColumn("Model", width="small"),
            "products": st.column_config.NumberColumn("Products", width="small"),
            "urls": st.column_config.NumberColumn("URLs", width="small"),
            "fetched": st.column_config.TextColumn(
                "Fetched", width="small", help="Pages fetched successfully / total."
            ),
            "results": st.column_config.NumberColumn("Rows", width="small"),
            "warnings": st.column_config.NumberColumn(
                "Warn", width="small", help="Failed fetches or extractions."
            ),
            "size_kb": st.column_config.NumberColumn(
                "Size", width="small", format="%.0f kB"
            ),
        },
    )

    # Selecting a row picks a run; otherwise fall back to the loaded one, then the newest.
    sel = list(event.selection.rows) if event and event.selection else []
    if sel:
        row = runs[sel[0]]
    elif active and any(r["run_id"] == active for r in runs):
        row = next(r for r in runs if r["run_id"] == active)
    else:
        row = runs[0]
    picked = row["run_id"]

    st.divider()

    if row["broken"]:
        st.error(
            f"**{picked}** cannot be read ({row['error']}). It may have been truncated "
            "by an interrupted save. Delete it to clear it from this list."
        )
        if st.button("Delete", key="runs_delete_broken"):
            delete_run(picked)
            st.rerun()
        return

    head, actions = st.columns([3, 2])
    with head:
        st.markdown(f"**{picked}**")
        bits = [
            STAGE_LABELS.get(row["stage"], row["stage"]),
            f"{row['urls']} URL(s) over {row['products']} product(s)",
            f"schema `{row['schema'] or '?'}`",
            f"{row['provider'] or '?'}/{row['model'] or '?'}",
        ]
        st.caption(" · ".join(bits))
        if picked == active:
            st.caption("Currently loaded in this session.")

    with actions:
        a, b = st.columns(2)
        if a.button(
            "Reload" if picked == active else "Load",
            type="primary",
            width=STRETCH,
            key="runs_load",
        ):
            try:
                state = load_run(picked)
            except Exception as e:
                st.error(f"Could not load: {e}")
            else:
                apply_loaded_run(state, picked)
                # Review and Export render before this tab, so without a rerun they would
                # keep showing the previous run until the next interaction.
                st.rerun()

        # Deleting is irreversible, so it sits behind a confirmation rather than firing
        # on a single stray click next to the Load button.
        with b.popover("Delete", width=STRETCH):
            st.markdown(f"Delete **{picked}** from disk?")
            st.caption("This cannot be undone.")
            if st.button("Delete permanently", key="runs_delete_confirm"):
                delete_run(picked)
                if picked == active:
                    st.session_state.active_run_id = ""
                st.rerun()

    if row["stage"] != "reconciled":
        st.info(
            f"This run stopped at **{STAGE_LABELS.get(row['stage'], row['stage'])}**, so "
            "it has no finished dataset. Load it and pick the matching option in the Run "
            "tab to continue from where it left off."
        )
    if row["warnings"]:
        st.caption(
            f"{row['warnings']} warning(s) recorded — "
            f"{row['pages_failed']} page(s) failed to fetch. Details after loading."
        )


# ---------------------------------------------------------------------- storage tab


BACKEND_LABELS = {
    "files": "JSON files only",
    "db": "Database only",
    "both": "Both (files + database)",
}


def tab_storage() -> None:
    from pipeline.db import DEFAULT_SQLITE_PATH_URL, redact, sqlite_path

    st.subheader("Storage")
    st.caption(
        "Runs are written as JSON files by default. Point them at a database instead to "
        "query them with SQL, share them between machines, or survive a container with "
        "no persistent disk."
    )

    s = get_settings()

    with st.form("storage_form"):
        backend = st.radio(
            "Where runs are saved",
            BACKENDS,
            index=BACKENDS.index(s.backend),
            format_func=lambda b: BACKEND_LABELS[b],
            horizontal=True,
            help="'Both' keeps writing JSON while you trial a database — if the database "
            "is unreachable the run is still safe on disk.",
        )
        db_url = st.text_input(
            "Database URL",
            value=s.db_url,
            placeholder=DEFAULT_SQLITE_PATH_URL,
            help="Any SQLAlchemy URL. Leave blank for the built-in SQLite file. "
            "PostgreSQL: postgresql+psycopg://user:pw@host:5432/dbname · "
            "MySQL: mysql+pymysql://user:pw@host:3306/dbname",
        )
        saved = st.form_submit_button("Save settings", type="primary")

    if saved:
        set_settings(backend, db_url)
        st.success(f"Saved. New runs go to: {BACKEND_LABELS[backend]}.")
        st.rerun()

    if not s.uses_db:
        st.info(
            "Database saving is off. Choose **Database only** or **Both** above, then "
            "use the migration tools below to copy existing runs across."
        )

    st.divider()

    # ── connection ───────────────────────────────────────────────
    st.markdown("**Connection**")
    url = s.resolved_url()
    st.code(redact(url), language="text")

    path = sqlite_path(url)
    if path is not None:
        st.caption(
            f"SQLite file: `{path}` — "
            + (
                f"{path.stat().st_size / 1_000_000:.1f} MB"
                if path.exists()
                else "not created yet"
            )
        )

    c1, c2 = st.columns(2)
    if c1.button("Test connection", width=STRETCH, key="db_test"):
        ok, msg = _db_check(url)
        (st.success if ok else st.error)(msg)
        if not ok and ("No module named" in msg or "Can't load plugin" in msg):
            st.caption(
                "The driver for this database is not installed. Install it into the "
                "same environment — `psycopg[binary]` for PostgreSQL, `PyMySQL` for "
                "MySQL — then test again."
            )
    if c2.button("Refresh stats", width=STRETCH, key="db_stats_btn"):
        st.session_state["db_stats"] = _db_stats(url)

    stats = st.session_state.get("db_stats")
    if stats:
        if "error" in stats:
            st.error(stats["error"])
        else:
            st.dataframe(
                pd.DataFrame([{"table": k, "rows": v} for k, v in stats.items()]),
                width=STRETCH,
                hide_index=True,
            )

    st.divider()

    # ── migration ────────────────────────────────────────────────
    st.markdown("**Move runs between backends**")
    st.caption(
        "Copying never deletes the source, so it is safe to run twice — existing runs "
        "are skipped unless you tick overwrite."
    )
    overwrite = st.checkbox(
        "Overwrite runs that already exist at the destination", key="db_overwrite"
    )

    m1, m2 = st.columns(2)
    if m1.button("Import JSON files → database", width=STRETCH, key="db_import"):
        states, errors = load_all_run_files()
        if not states and not errors:
            st.info("No JSON runs found in `runs/`.")
        else:
            try:
                report = _get_db().import_states(states, overwrite=overwrite)
            except Exception as e:  # noqa: BLE001 - shown to the user
                st.error(f"Import failed: {e}")
            else:
                st.success(
                    f"Imported {report['imported']}, skipped {report['skipped']}, "
                    f"failed {report['failed']}."
                )
                for err in errors + report["errors"]:
                    st.caption(f"⚠️ {err}")
                st.cache_data.clear()

    if m2.button("Export database → JSON files", width=STRETCH, key="db_export_files"):
        try:
            db = _get_db()
            written = 0
            for state in db.export_states():
                if not overwrite and (RUNS_DIR / f"{state.run_id}.json").exists():
                    continue
                save_run_file(state)
                written += 1
        except Exception as e:  # noqa: BLE001 - shown to the user
            st.error(f"Export failed: {e}")
        else:
            st.success(f"Wrote {written} run(s) to `runs/`.")
            st.cache_data.clear()

    st.divider()

    # ── download / restore ───────────────────────────────────────
    st.markdown("**Backup and restore**")
    d1, d2 = st.columns(2)

    with d1:
        if path is not None and path.exists():
            try:
                blob = _get_db().file_bytes()
            except Exception as e:  # noqa: BLE001 - shown to the user
                st.error(f"Could not read the database file: {e}")
                blob = None
            if blob:
                st.download_button(
                    "Download SQLite file",
                    blob,
                    file_name=path.name,
                    mime="application/vnd.sqlite3",
                    width=STRETCH,
                    key="db_download_file",
                )
        else:
            st.caption("Download of the raw file is only available for SQLite.")

    with d2:
        if st.button("Prepare JSON dump", width=STRETCH, key="db_dump_prepare"):
            try:
                st.session_state["db_dump"] = _get_db().dump_json()
            except Exception as e:  # noqa: BLE001 - shown to the user
                st.error(f"Dump failed: {e}")
        dump = st.session_state.get("db_dump")
        if dump:
            st.download_button(
                "Download JSON dump",
                dump.encode("utf-8"),
                file_name=f"runs_dump_{datetime.now():%Y%m%d_%H%M%S}.json",
                mime="application/json",
                width=STRETCH,
                key="db_dump_download",
            )

    uploaded = st.file_uploader(
        "Restore from a JSON dump or a single run file",
        type=["json"],
        key="db_restore",
        help="Accepts a dump produced above, or one exported run JSON.",
    )
    if uploaded is not None and st.button(
        "Restore into the current backend", key="db_restore_go"
    ):
        try:
            states = _parse_restore(uploaded.getvalue())
        except Exception as e:  # noqa: BLE001 - shown to the user
            st.error(f"Could not read that file: {e}")
        else:
            report = {"imported": 0, "skipped": 0, "failed": 0, "errors": []}
            if s.uses_db:
                report = _get_db().import_states(states, overwrite=overwrite)
            if s.uses_files:
                for state in states:
                    if overwrite or not (RUNS_DIR / f"{state.run_id}.json").exists():
                        save_run_file(state)
            st.success(
                f"Restored {len(states)} run(s) from the file "
                f"(database: {report['imported']} imported, {report['skipped']} skipped)."
            )
            st.cache_data.clear()


def _get_db():
    """A RunDB for the configured URL even when the backend is set to files only."""
    from pipeline.db import RunDB

    return get_db() or RunDB(get_settings().resolved_url())


@st.cache_data(show_spinner=False, ttl=30)
def _db_check(url: str) -> tuple[bool, str]:
    from pipeline.db import RunDB

    return RunDB(url).check()


def _db_stats(url: str) -> dict:
    from pipeline.db import RunDB

    try:
        return RunDB(url).stats()
    except Exception as e:  # noqa: BLE001 - rendered as an error row
        return {"error": str(e)}


def _parse_restore(blob: bytes) -> list[RunState]:
    raw = json.loads(blob.decode("utf-8-sig"))
    items = raw.get("runs") if isinstance(raw, dict) and "runs" in raw else [raw]
    if not isinstance(items, list):
        raise ValueError("Expected a run object or a {'runs': [...]} dump.")
    return [RunState.model_validate(item) for item in items]


# --------------------------------------------------------------------- advanced tab


def tab_advanced() -> None:
    st.subheader("Advanced")
    st.caption(
        "The system prompts sent to the model at every LLM stage. They are split by "
        "scenario because an alternatives article is written in two registers: the "
        "sections about the subject product, and the per-tool sections comparing it "
        "against alternatives."
    )

    ss = st.session_state
    spec = current_spec()

    # Open on whatever the loaded schema actually uses; showing the other scenario's
    # prompts first is a reliable way to edit the wrong ones.
    if spec and ss.get("_prompt_spec_nonce") != ss.spec_nonce:
        ss._prompt_spec_nonce = ss.spec_nonce
        ss.prompt_scenario = spec.scenario.value

    scenarios = [s.value for s in Scenario]
    picked = st.radio(
        "Scenario",
        scenarios,
        horizontal=True,
        key="prompt_scenario",
        format_func=lambda v: SCENARIO_LABELS[Scenario(v)],
    )

    if spec:
        if spec.scenario.value == picked:
            st.success(
                f"Schema **{spec.name}** uses this scenario, so these prompts apply to "
                "the next run.",
                icon="✅",
            )
        else:
            st.info(
                f"Schema **{spec.name}** is set to **{spec.scenario.value}**, so edits "
                "here are saved but will not affect the next run. Change the scenario "
                "in the Schema tab to use them.",
                icon="ℹ️",
            )

    if picked == Scenario.TOOLS.value:
        st.caption(
            "The tools prompts are stricter about two things: never attributing a "
            "competitor's traits to the subject (most sources are 'A vs B' posts), and "
            "never converting, rounding or averaging a price."
        )
    else:
        st.caption(
            "The general prompts cover the article's subject product — company history, "
            "what the platform is, who uses it, and where it falls short."
        )

    saved = get_prompts(picked)
    defaults = default_prompts(picked)
    drafts: dict = ss.prompt_drafts

    edited: dict[str, str] = {}
    for stage in STAGES:
        draft_key = f"{picked}:{stage}"
        current = drafts.get(draft_key) or getattr(saved, stage)
        with st.expander(
            PROMPT_STAGE_LABELS[stage],
            expanded=(stage == "extract"),
        ):
            marks = []
            if is_customised(picked, stage):
                marks.append("saved override")
            if drafts.get(draft_key):
                marks.append("unsaved edit")
            if marks:
                st.caption(" · ".join(marks))

            text = st.text_area(
                "System prompt",
                value=current,
                height=320,
                key=f"prompt_text_{picked}_{stage}_{ss.prompt_nonce}",
                label_visibility="collapsed",
            )
            edited[stage] = text

            if text.strip() != getattr(defaults, stage).strip():
                with st.popover("Show built-in default"):
                    st.code(getattr(defaults, stage), language="text")

    c1, c2, c3 = st.columns(3)
    if c1.button("Save prompts", type="primary", width=STRETCH, key="prompts_save"):
        save_prompts(picked, PromptSet(**edited))
        # Saved text is no longer a draft, or the two would fight over precedence.
        for stage in STAGES:
            drafts.pop(f"{picked}:{stage}", None)
        ss.prompt_nonce += 1
        st.success("Saved. New runs will use these prompts.")
        st.rerun()

    if c2.button(
        "Use without saving",
        width=STRETCH,
        key="prompts_try",
        help="Applies the text above to runs in this session only.",
    ):
        for stage in STAGES:
            if edited[stage].strip() == getattr(saved, stage).strip():
                drafts.pop(f"{picked}:{stage}", None)
            else:
                drafts[f"{picked}:{stage}"] = edited[stage].strip()
        st.success("Applied to this session.")
        st.rerun()

    if c3.button("Reset to defaults", width=STRETCH, key="prompts_reset"):
        reset_prompts(picked)
        for stage in STAGES:
            drafts.pop(f"{picked}:{stage}", None)
        ss.prompt_nonce += 1
        st.success("Restored the built-in prompts.")
        st.rerun()

    st.caption(
        "Prompts are part of every cache key, so an edited prompt correctly misses the "
        "cache and re-runs. Unchanged fields still cost nothing."
    )


# ------------------------------------------------------------------------- help tab


def tab_help() -> None:
    st.subheader("How this app works")

    st.markdown("""
        This tool turns reference URLs into a **structured comparison dataset** —
        one row per product, with every claim traceable back to the page it came from.

        ### Start here

        The normal path is four clicks, because the brief already contains the URLs:

        1. **Schema** → *Import a brief workbook* → pick a sheet → **Import sheet**.
           The General and Tools sheets become separate schemas; import them one at a
           time.  Press **Save to disk** to keep it.
        2. **Input** → type the product name → **Seed input from schema**.  This builds
           the URL list from the brief's own *Source urls*, so there is no sheet to
           prepare.
        3. **Run** → check the preflight (calls, rough cost, any field with no source)
           → **Start run**.
        4. **Review** the conflicts, then **Export** the workbook.

        Everything else on this page is detail you only need when something looks wrong.

        ### The pipeline (5 stages)

        **1. Scrape** — Fetch every URL.  A fast static HTTP check runs first; most
        pages (docs, blog posts) skip the heavy Chromium browser.  Only JS-rendered
        pages (pricing tables, SPAs) launch headless Chrome.  Evidence you pasted as
        *custom input* is already text, so it skips this stage entirely.

        **2. Extract** — Each source goes to the LLM with a prompt listing only the
        fields that source was nominated for.  The model returns found/not-found for
        every field, plus a verbatim supporting quote.  Quotes are checked against the
        source text — unverifiable ones are flagged as possible inventions.

        **3. Reconcile** — Claims from multiple sources about the same field are merged
        into one value.  The LLM consolidates duplicates, preserves distinct points, and
        flags genuine contradictions (e.g. two different starting prices).  A mechanical
        fallback catches cases where the LLM returns nothing.

        **3b. Web check** *(optional)* — Flagging a conflict says the sources disagree,
        not who is right, and the scraped pages cannot settle it — they *are* the
        disagreement.  Switch on **Resolve conflicts with web search** in the sidebar and
        each conflicting field goes to the model's hosted search tool, which checks live
        pages (the vendor's own site first) and either settles it with citations or says
        it could not.

        **4. Review** — Inspect conflicts, trace any cell back to its source URLs and
        individual claims, and check coverage (which fields are empty, which rely on a
        single source).

        **5. Export** — Download a 7-sheet Excel workbook: Dataset, Transposed,
        QA coverage, Provenance, Claims, Pages, and the Schema brief.

        ### Tabs

        | Tab | What you do |
        |---|---|
        | **Schema** | Edit the content brief — fields, questions, source URLs and pasted evidence.  Saved as YAML. |
        | **Input** | Usually just *Seed input from schema*.  Or upload a sheet with Product, Key and URL columns. |
        | **Run** | Choose stages, see the preflight (cost + coverage), and start the pipeline. |
        | **Review** | Inspect conflicts, trace provenance, check coverage. |
        | **Export** | Download the workbook. |
        | **Saved runs** | Reopen past runs — every stage is persisted to disk. |
        | **Advanced** | Edit the system prompts sent at every LLM stage, per scenario. |
        | **Storage** | Choose where runs are saved (JSON files and/or a database), migrate, back up, and restore. |

        ### Scenarios and prompts (Advanced tab)

        An alternatives article is written in two registers, so the prompts come in two
        sets and a schema declares which one it uses.

        - **General** — the sections about the article's *subject* product: company
          history, what the platform is, who uses it, where it falls short.
        - **Tools** — the per-alternative sections (strengths, limitations, pricing).
          These prompts are stricter about two things: never attributing a competitor's
          traits to the subject (most sources are "A vs B" posts), and never converting,
          rounding or averaging a price.

        Each scenario has a prompt for all three LLM stages — extraction, merge, and web
        check.  **Save prompts** writes them to `prompts.json`; **Use without saving**
        applies them to this session only, which is the cheap way to trial a change.
        Prompts are part of every cache key, so an edit correctly misses the cache.

        ### Importing a brief workbook

        The Schema tab reads the client's data-structure spreadsheet directly (columns
        *Article - H2*, *Article - H3*, *Key*, *Prompt*, *Anchors*, *Source urls*,
        *Custom input*).  Each sheet becomes one schema, and the scenario is taken from
        the sheet name, so import the General and Tools sheets separately.

        - **H2/H3 are merged cells**, so blanks inherit from the row above.  H2 is the
          article section, H3 the column label.
        - **Anchors** are angles the brief insists on covering; they reach the model as
          "Must cover".
        - **Custom input** is evidence you pasted by hand.  It becomes a source of its
          own — like one more URL, but with no fetch — so its claims are merged and
          traced exactly like a scraped page.
        - **Source urls** decide routing.  Press *Seed input from schema* in the Input
          tab and a run can start without building a sheet at all.

        ### Routing: which source answers which field

        The brief lists source URLs per key, and that mapping is **many-to-many** — one
        key cites several URLs, and the same URL is cited by several keys.

        - A URL is **fetched once and extracted once**, and is asked only the questions
          it was listed under.  Listing one URL under five keys costs one call, not five.
        - A URL that is **not** in the brief (pasted in by hand with no Key column) is
          tried against every field, so nothing is silently skipped.
        - Field-specific instructions belong in the field's own **question**, **guidance**
          and **anchors**.  The master prompts in the Advanced tab deliberately hold only
          rules that apply to every field.

        ### Storage

        Every run is written as a JSON file under `runs/` by default. The **Storage** tab
        switches this to a **database** — a local SQLite file (`data/runs.db`) or any
        server SQLAlchemy supports — and can copy runs between backends.

        - **No login.**  Runs are not tied to an account or a browser session.  They are
          files on the machine running the app, so they survive a refresh, a browser
          restart and a reboot.  The **Saved runs** tab prints the exact folder.
        - **Backends** — `files`, `db`, or `both`.  `both` writes to JSON and the
          database at once, so a database outage can never lose a run.
        - **Databases** — Leave the URL blank for SQLite, or use a SQLAlchemy URL for a
          shared server: `postgresql+psycopg://user:pw@host:5432/dbname` or
          `mysql+pymysql://user:pw@host:3306/dbname`.  Servers need their driver
          installed (`psycopg[binary]`, `PyMySQL`).
        - **Queryable** — Each run is stored as its exact JSON payload (lossless reload)
          plus flattened tables — `run_inputs`, `run_pages`, `run_extractions`,
          `run_claims`, `run_results`, `run_fields`, `run_warnings` — so any SQL client
          can read the data directly.
        - **Migrate / back up** — Import existing JSON runs into the database, export
          them back to files, download the SQLite file, or download a JSON dump of every
          run and restore from it.  Copying never deletes the source.
        - **Persistent** — Your choice is remembered in `storage.json`.  In Docker, the
          `data/` volume keeps the database across rebuilds.

        ### Sidebar

        - **Browser** — Install Chromium if missing (only needed on first deploy or
          after an idle shutdown on Streamlit Cloud).
        - **Model** — Pick your LLM provider and model.  Each provider remembers its
          own API key, endpoint, and model, so switching is safe.
        - **Resolve conflicts with web search** — Fact-check disagreements against live
          web pages.  See below.
        - **Parameters** — Temperature, token budget, concurrency, cache toggle.

        ### Web conflict resolution

        Enabled with the sidebar toggle, this adds a pass after merging that fact-checks
        every field where sources disagree.

        - **Only OpenAI and Anthropic** host a web search tool, so the toggle is disabled
          for DeepSeek and Ollama.
        - **Only conflicting fields** are checked, so the cost scales with disagreement,
          not with schema size.  Each search costs roughly $0.01 on top of tokens; cap
          them per field with *Max searches per conflict*.
        - **Three outcomes.**  *Resolved* replaces the value and records the reasoning,
          the citations, and what the value was before.  *Unresolved* leaves the merged
          value untouched and says so.  *Failed* records the error — the run continues
          either way.
        - **The conflict flag stays on** even when a check succeeds: the sources really
          did disagree, and a reviewer should see that the cell was arbitrated rather
          than agreed.
        - **Verdicts are cached**, saved with the run, shown in Review, and exported in
          the Provenance sheet.  *Web-check conflicts only* in the Run tab re-checks a
          finished run without repeating the merge.

        ### Tips

        - **Tune prompts before spending.**  Change a field's question or guidance in the
          Schema tab, re-run extraction only, and the cache ensures only changed fields
          re-run.
        - **Put field rules on the field.**  If only one column needs a rule (how to
          quote a price, which licence to name), write it in that field's *question*,
          *guidance* or *anchors* — not in the master prompt, which every field sees.
        - **Paste evidence you don't want scraped.**  A Reddit reply or a note from a
          call goes in the field's *custom input* and is treated as a source of its own.
        - **A field with no source stays empty.**  The Schema tab warns about these
          before you spend anything; give the field a URL or some custom input.
        - **Review conflicts first.**  Fields marked `[CONFLICT]` are where errors
          concentrate — two sources gave incompatible facts.
        - **Let the web settle stale figures.**  Pricing and integration counts are the
          usual culprits, and blog posts go out of date.  The web check reads the
          vendor's own page, so it is the cheapest way to fix those cells.
        - **Single-source fields are unverified.**  If only one page covered a field,
          there is no cross-check.  Verify those cells manually.
        - **The cache is your friend.**  Re-running an unchanged pipeline costs nothing
          (no API calls, no fetches).  Clear it if page content has changed.
        - **Static fallback is automatic.**  Even if Chromium isn't installed, most
          pages still work via plain HTTP extraction — only JS-heavy pricing tables
          need the browser.
        - **Back up before you clear runs.**  Use the Storage tab to download the
          SQLite file or a JSON dump — deleting a run is permanent.
        """)


# ----------------------------------------------------------------------------- main


def main() -> None:
    init_state()
    render_sidebar()

    st.title("SEO Content Pipeline | Mike - RINGHEL")
    st.caption(
        "Reference URLs in, a structured comparison dataset out — with every claim "
        "traceable to the page it came from.  Built by Ali."
    )

    t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs(
        [
            "Schema",
            "Input",
            "Run",
            "Review",
            "Export",
            "Saved runs",
            "Advanced",
            "Storage",
            "Help",
        ]
    )
    with t1:
        tab_schema()
    with t2:
        tab_input()
    with t3:
        tab_run()
    with t4:
        tab_review()
    with t5:
        tab_export()
    with t6:
        tab_runs()
    with t7:
        tab_advanced()
    with t8:
        tab_storage()
    with t9:
        tab_help()


if __name__ == "__main__":
    main()
