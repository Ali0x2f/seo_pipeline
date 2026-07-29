"""SEO content pipeline - Streamlit front end.

Scrape reference pages, extract structured claims per source, reconcile them into one
row per product, review provenance, export.
"""

from __future__ import annotations

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


from config import SCHEMA_DIR, Config
from jobs import start_job
from pipeline import cache as cache_mod
from pipeline import exporter
from pipeline.models import RunState
from pipeline.providers import (
    PROVIDERS,
    SUGGESTED_MODELS,
    build_provider,
    estimate_cost,
)
from pipeline.runner import parse_input, preflight, run_pipeline
from pipeline.schema import (
    FieldShape,
    FieldSpec,
    FillFrom,
    ListOutput,
    SchemaSpec,
    list_schemas,
    load_schema_by_name,
)
from pipeline.store import delete_run, list_runs, load_run, runs_signature

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

# Node names routinely contain commas (e.g. "Strengths, limitations, best for"), so the
# editor separates lists with a pipe. Commas here would silently shred such names.
NODE_SEP = "|"


def fields_to_df(spec: SchemaSpec) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "key": f.key,
                "label": f.label,
                "question": f.question,
                "shape": f.shape.value,
                "nodes": f" {NODE_SEP} ".join(f.nodes),
                "max_items": f.max_items,
                "guidance": f.guidance,
                "source": f.fill_from.value,
            }
            for f in spec.fields
        ]
    )


def df_to_fields(df: pd.DataFrame) -> tuple[list[FieldSpec], list[str]]:
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
                    nodes=[
                        n.strip()
                        for n in str(r.get("nodes") or "").split(NODE_SEP)
                        if n.strip()
                    ],
                    max_items=int(r.get("max_items") or 10),
                    guidance=str(r.get("guidance") or "").strip(),
                    fill_from=FillFrom(str(r.get("source") or "extract").strip()),
                )
            )
        except Exception as e:
            errors.append(f"Row {i + 1} ({key}): {e}")
    return fields, errors


def tab_schema() -> None:
    st.subheader("Output schema")
    st.caption(
        "The deliverable's shape lives here as data, not in code. Each row is one output "
        "column: the question that defines it, the shape of the answer, and which input "
        "nodes are allowed to answer it."
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
            st.success(f"Loaded {picked}")
            st.rerun()

    spec = current_spec()
    if spec is None:
        st.info("No schema available. Create one below.")
        return

    a, b, c = st.columns(3)
    a.text_input("Schema name", value=spec.name, key="spec_name")
    b.text_input(
        "Entity label",
        value=spec.entity_label,
        key="spec_entity",
        help="What one row represents. 'Product' here.",
    )
    c.selectbox(
        "List rendering",
        [o.value for o in ListOutput],
        index=[o.value for o in ListOutput].index(spec.list_output.value),
        key="spec_listout",
        help="How list fields are written into a cell.",
    )
    st.text_area(
        "Nodes (one per line)",
        value="\n".join(spec.nodes),
        key="spec_nodes",
        height=90,
        help="The Node values your input sheet uses, exactly as spelled there. One per "
        "line, because node names often contain commas.",
    )

    st.markdown("**Fields**")
    edited = st.data_editor(
        fields_to_df(spec),
        num_rows="dynamic",
        width=STRETCH,
        key="field_editor",
        column_config={
            "key": st.column_config.TextColumn(
                "key",
                help="snake_case identifier. Changing it invalidates cached "
                "extractions for the field.",
                width="small",
            ),
            "label": st.column_config.TextColumn(
                "label", help="Exact spreadsheet header."
            ),
            "question": st.column_config.TextColumn(
                "question",
                help="Defines the field. This text is sent to the model.",
                width="large",
            ),
            "shape": st.column_config.SelectboxColumn(
                "shape", options=[s.value for s in FieldShape], width="small"
            ),
            "nodes": st.column_config.TextColumn(
                "nodes",
                help="Separate multiple nodes with a pipe ( | ), since node "
                "names often contain commas. Empty means every node may "
                "answer this field.",
            ),
            "max_items": st.column_config.NumberColumn(
                "max", min_value=1, max_value=50, width="small"
            ),
            "guidance": st.column_config.TextColumn("guidance", width="large"),
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
        fields, errors = df_to_fields(edited)
        if errors:
            for e in errors:
                st.error(e)
        else:
            try:
                new_spec = SchemaSpec(
                    name=st.session_state.spec_name.strip() or spec.name,
                    entity_label=st.session_state.spec_entity.strip() or "Product",
                    description=spec.description,
                    nodes=[
                        n.strip()
                        for n in st.session_state.spec_nodes.splitlines()
                        if n.strip()
                    ],
                    list_output=ListOutput(st.session_state.spec_listout),
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

    with st.expander("Node → field routing"):
        for node in spec.all_nodes():
            fs = spec.fields_for_node(node)
            labels_html = (
                "".join(
                    f'<span style="display:inline-block;background:#2d3748;color:#e2e8f0;'
                    f"border-radius:4px;padding:1px 8px;margin:2px 4px 2px 0;"
                    f'font-size:0.85em;white-space:nowrap;">{f.label}</span>'
                    for f in fs
                )
                or '<span style="color:#718096;font-style:italic;">(none)</span>'
            )
            st.markdown(
                f'<div style="margin:4px 0;">'
                f'<span style="font-weight:600;color:#e2e8f0;margin-right:8px;">{node}</span>'
                f"{labels_html}</div>",
                unsafe_allow_html=True,
            )

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
        "One row per URL: which product it describes and which node (section) it feeds. "
        "Columns are matched loosely, and the URL column is detected by content."
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
            placeholder="Product\tNode\tURL\nn8n\tPricing\thttps://n8n.io/pricing/",
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

    df = pd.DataFrame([r.model_dump() for r in rows])
    m1, m2, m3 = st.columns(3)
    m1.metric("URLs", len(rows))
    m2.metric("Products", df["product"].nunique())
    m3.metric("Nodes", df["node"].nunique())
    st.dataframe(df, width=STRETCH, hide_index=True)

    if spec is None:
        st.warning("Load a schema to run preflight checks.")
        return

    st.markdown("**Preflight**")
    pf = preflight(rows, spec, build_cfg())
    st.session_state.preflight = pf
    if pf["problems"]:
        for p in pf["problems"]:
            st.warning(p)
    else:
        st.success("Every schema field is fed by at least one URL.")
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

    opts = {
        "Everything (scrape → extract → merge)": ("scrape", "extract", "reconcile"),
        "Re-extract and merge (reuse fetched pages)": ("extract", "reconcile"),
        "Re-merge only (reuse extractions)": ("reconcile",),
    }
    choice = st.radio("Stages", list(opts), horizontal=False, key="run_stages")
    stages = opts[choice]

    prior: RunState | None = st.session_state.run
    if stages != ("scrape", "extract", "reconcile") and prior is None:
        st.warning(
            "No previous run in this session; the full pipeline will run instead."
        )
        stages = ("scrape", "extract", "reconcile")

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
                state=prior if stages != ("scrape", "extract", "reconcile") else None,
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

        total_in = total_out = 0
        for name, u in (st.session_state.usages or {}).items():
            total_in += u.prompt_tokens
            total_out += u.completion_tokens
            st.caption(
                f"{name}: {u.calls} live call(s), {u.cached} cached, {u.errors} error(s), "
                f"{u.prompt_tokens:,} in / {u.completion_tokens:,} out tokens"
            )
        actual = estimate_cost(state.model, total_in, total_out)
        if actual:
            st.caption(f"Approximate spend this run: **${actual:.3f}**")

        if state.warnings:
            with st.expander(f"{len(state.warnings)} warning(s)"):
                for w in state.warnings:
                    st.write(f"- {w}")

    if st.button("Reset run state", key="run_reset"):
        st.session_state.job = None
        st.rerun()


# ---------------------------------------------------------------------- review tab


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
        st.warning(
            f"{len(conflicts)} field(s) where sources disagree. These need a human "
            "decision before publishing."
        )
        for product, label, rf in conflicts:
            with st.expander(f"{product} · {label}"):
                st.write(rf.conflict_note or "(no note)")
                st.text_area(
                    "Merged value",
                    rf.value,
                    height=100,
                    key=f"cf_{product}_{label}",
                    disabled=True,
                )
                st.caption("Sources: " + ", ".join(rf.sources))

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
        with st.expander(f"{ext.url[:95]}  ·  node: {ext.node}"):
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

    runs = _runs_index(runs_signature())
    if not runs:
        st.info(
            "No saved runs yet. Start one from the Run tab — each stage is saved as it "
            "completes."
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


# ------------------------------------------------------------------------- help tab


def tab_help() -> None:
    st.subheader("How this app works")

    st.markdown("""
        This tool turns reference URLs into a **structured comparison dataset** —
        one row per product, with every claim traceable back to the page it came from.

        ### The pipeline (5 stages)

        **1. Scrape** — Fetch every URL.  A fast static HTTP check runs first; most
        pages (docs, blog posts) skip the heavy Chromium browser.  Only JS-rendered
        pages (pricing tables, SPAs) launch headless Chrome.

        **2. Extract** — Each scraped page goes to the LLM with a prompt listing the
        schema fields for that page's node.  The model returns found/not-found for every
        field, plus a verbatim supporting quote.  Quotes are checked against the page
        text — unverifiable ones are flagged as possible inventions.

        **3. Reconcile** — Claims from multiple sources about the same field are merged
        into one value.  The LLM consolidates duplicates, preserves distinct points, and
        flags genuine contradictions (e.g. two different starting prices).  A mechanical
        fallback catches cases where the LLM returns nothing.

        **4. Review** — Inspect conflicts, trace any cell back to its source URLs and
        individual claims, and check coverage (which fields are empty, which rely on a
        single source).

        **5. Export** — Download a 7-sheet Excel workbook: Dataset, Transposed,
        QA coverage, Provenance, Claims, Pages, and the Schema brief.

        ### Tabs

        | Tab | What you do |
        |---|---|
        | **Schema** | Edit the content brief — add/remove fields, change questions, assign nodes.  Saved as YAML. |
        | **Input** | Paste a CSV or upload a file.  One row per URL with Product, Node, and URL columns. |
        | **Run** | Choose stages, see the preflight (cost + coverage), and start the pipeline. |
        | **Review** | Inspect conflicts, trace provenance, check coverage. |
        | **Export** | Download the workbook. |
        | **Saved runs** | Reopen past runs — every stage is persisted to disk. |

        ### Sidebar

        - **Browser** — Install Chromium if missing (only needed on first deploy or
          after an idle shutdown on Streamlit Cloud).
        - **Model** — Pick your LLM provider and model.  Each provider remembers its
          own API key, endpoint, and model, so switching is safe.
        - **Parameters** — Temperature, token budget, concurrency, cache toggle.

        ### Tips

        - **Tune prompts before spending.**  Change a field's question or guidance in the
          Schema tab, re-run extraction only, and the cache ensures only changed fields
          re-run.
        - **Review conflicts first.**  Fields marked `[CONFLICT]` are where errors
          concentrate — two sources gave incompatible facts.
        - **Single-source fields are unverified.**  If only one page covered a field,
          there is no cross-check.  Verify those cells manually.
        - **The cache is your friend.**  Re-running an unchanged pipeline costs nothing
          (no API calls, no fetches).  Clear it if page content has changed.
        - **Static fallback is automatic.**  Even if Chromium isn't installed, most
          pages still work via plain HTTP extraction — only JS-heavy pricing tables
          need the browser.
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

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(
        ["Schema", "Input", "Run", "Review", "Export", "Saved runs", "Help"]
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
        tab_help()


if __name__ == "__main__":
    main()
