"""Promo Finder: Streamlit front end.

Two ways in:
  1. Run: upload a show's episode screenplays (.txt files or one .zip), add an API key, run.
  2. View: upload a promos.json from an earlier run to browse it (no key needed).
No show data is bundled with the app (the screenplays are confidential).
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile

import altair as alt
import pandas as pd
import streamlit as st

from promo import pipeline
from promo.llm import LLM, PRESETS, LLMError, load_dotenv
from promo.parse import load_show

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")

TIER_COLORS = {"acquisition": "#2a78d6", "re-engagement (spoils the plot)": "#eb6834", "weak": "#a3a29c"}
# blue / orange = categorical slots 1-2 of the reference palette; weak = muted neutral (de-emphasised)

st.title("🎬 Promo Finder")
st.caption("Reads a microdrama's screenplays and proposes 20-60s promo clips: a hook in the first 3 seconds, "
           "a cliffhanger at the cut, understandable to a stranger. Every clip gets exact episode timestamps, "
           "an independent 'stranger' QA score and a posting order.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model")
    preset = st.selectbox("Provider", list(PRESETS), index=0,
                          help="Any OpenAI-compatible endpoint works; presets fill in the URL and default models.")
    base, gen_chain, judge_chain = PRESETS[preset]
    key = st.text_input("API key", type="password",
                        help="Free Gemini key: aistudio.google.com/apikey. Used for this run only, never stored.")
    # A server-side key (env / host secret) is used if the box is empty; it is never sent to the browser.
    load_dotenv()
    if not key and os.environ.get("LLM_API_KEY"):
        key = os.environ["LLM_API_KEY"]
        st.caption("Using the server's configured key.")
    gen_models = st.text_input("Generator models (fallback order)", ",".join(gen_chain))
    judge_models = st.text_input("Blind judge models (use different ones)", ",".join(judge_chain))
    use_audit = st.checkbox("Show-aware audit (stronger model)", value=True,
                            help="A second opinion that knows the whole story: catches spoilers the blind judge can't.")
    audit_models = st.text_input("Auditor models", ",".join(gen_chain), disabled=not use_audit)
    chunk = st.slider("Episodes per proposal call", 1, 10, pipeline.CHUNK,
                      help="Bigger = fewer calls (free tiers allow ~20/day per model) and more cross-episode context.")
    st.divider()
    st.markdown("**How it works**\n1. Parse script → timed blocks\n2. LLM proposes clips by *block id*\n"
                "3. Code computes timestamps, enforces 20-60s, never cuts a line in half\n"
                "4. A different LLM judges each clip blind, as a stranger\n"
                "5. A show-aware auditor re-scores and rates spoilers\n6. Code ranks into tiers; LLM explains")

tab_run, tab_view = st.tabs(["▶ Run on a show", "📂 View results"])


def _extract(files) -> str:
    d = tempfile.mkdtemp(prefix="show_")
    for f in files:
        data = f.getvalue()
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for n in z.namelist():
                    if n.lower().endswith(".txt") and not n.startswith("__MACOSX"):
                        with open(os.path.join(d, os.path.basename(n)), "wb") as out:
                            out.write(z.read(n))
        elif f.name.lower().endswith(".txt"):
            with open(os.path.join(d, os.path.basename(f.name)), "wb") as out:
                out.write(data)
    return d


with tab_run:
    files = st.file_uploader("Episode screenplays (.txt, or one .zip of them)", type=["txt", "zip"],
                             accept_multiple_files=True)
    c1, c2 = st.columns([1, 1])
    mode = c1.radio("Mode", ["Full (LLM)", "Rules-only baseline (no key)"], horizontal=True)
    ep_filter = c2.text_input("Episodes (optional)", placeholder="e.g. 1-10")
    if st.button("Find promos", type="primary", disabled=not files):
        data_dir = _extract(files)
        try:
            eps = load_show(data_dir)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not parse screenplays: {e}")
            st.stop()
        st.success(f"Parsed {len(eps)} episodes, {sum(len(e.blocks) for e in eps)} timed blocks, "
                   f"{sum(e.duration for e in eps) / 60:.0f} min of show.")
        subset = None
        if ep_filter.strip():
            from run import parse_eps
            subset = parse_eps(ep_filter.strip())
        out_dir = tempfile.mkdtemp(prefix="out_")
        pipeline.CHUNK = chunk
        llm = judge = auditor = None
        if mode.startswith("Full"):
            if not key:
                st.error("Add an API key in the sidebar, or use the rules-only baseline.")
                st.stop()
            split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
            cache = os.path.join(tempfile.gettempdir(), "promo_cache")
            llm = LLM("propose", split(gen_models), base_url=base, api_key=key, cache_dir=cache)
            judge = LLM("judge", split(judge_models), base_url=base, api_key=key, cache_dir=cache)
            if use_audit:
                auditor = LLM("audit", split(audit_models), base_url=base, api_key=key, cache_dir=cache)
        with st.status("Running…", expanded=True) as status:
            try:
                rows = pipeline.run(data_dir, out_dir, "llm" if llm else "baseline", llm, judge, subset,
                                    cb=lambda m: st.write(m), eps=eps,
                                    show_name=os.path.splitext(files[0].name)[0] if len(files) == 1 else None,
                                    audit_llm=auditor)
                status.update(label="Done", state="complete")
            except LLMError as e:
                status.update(label="Model error", state="error")
                st.error(str(e))
                st.stop()
        st.session_state["rows"] = rows
        st.session_state["stats"] = json.load(open(os.path.join(out_dir, "run_stats.json"), encoding="utf-8"))
        st.session_state["report"] = open(os.path.join(out_dir, "report.md"), encoding="utf-8").read()
        st.session_state["csv"] = open(os.path.join(out_dir, "promos.csv"), encoding="utf-8-sig").read()

with tab_view:
    up = st.file_uploader("promos.json from an earlier run (optionally also run_stats.json)", type=["json"],
                          accept_multiple_files=True, key="view")
    for f in up or []:
        blob = json.loads(f.getvalue())
        if isinstance(blob, list):
            st.session_state["rows"] = blob
        elif isinstance(blob, dict) and "summary" in blob:
            st.session_state["stats"] = blob

# ------------------------------------------------------------------ results
rows = st.session_state.get("rows")
if rows:
    st.divider()
    stats = st.session_state.get("stats") or {}
    summ = stats.get("summary") or pipeline.summarize(rows)
    m = st.columns(5)
    m[0].metric("Valid clips", summ["clips"])
    m[1].metric("Acquisition-ready", (summ.get("by_tier") or {}).get("acquisition", summ["qualifying"]),
                help="Hook ≥3 and cliffhanger ≥3 (of 5) and spoiler risk ≤3: safe to show strangers")
    m[2].metric("Episodes with a qualifying clip", summ["episodes_with_qualifying"])
    m[3].metric("Stitched / cross-episode", f"{summ['stitched']} / {summ['cross_episode']}")
    m[4].metric("Rejected by code checks", stats.get("n_rejected", "–"))

    df = pd.DataFrame([{
        "rank": r["rank"], "episode": r["episode"], "score": r["score"],
        "tier": r.get("tier") or ("acquisition" if r["qualifies"] else "weak"),
        "where": " + ".join(f"E{s['episode']} {s['start']}–{s['end']}" for s in r["segments"]),
        "duration_s": r["duration_s"], "hook": r["hook"], "cliffhanger": r["cliffhanger"],
        "judge_hook": r["judge"].get("hook"), "judge_cliff": r["judge"].get("cliffhanger"),
        "judge_standalone": r["judge"].get("standalone"), "spoiler": r["judge"].get("spoiler_risk"),
        "flag": r["judge"].get("content_flag"),
        "audit_hook": (r.get("audit") or {}).get("hook"), "audit_cliff": (r.get("audit") or {}).get("cliffhanger"),
        "audit_spoiler": (r.get("audit") or {}).get("spoiler_risk"),
    } for r in rows])

    st.subheader("Where the promos are")
    chart_df = df.dropna(subset=["score"])
    if not chart_df.empty:
        chart = (
            alt.Chart(chart_df)
            .mark_circle(size=90, opacity=0.85, stroke="white", strokeWidth=1.5)
            .encode(
                x=alt.X("episode:Q", title="Episode", scale=alt.Scale(domain=[0, int(df.episode.max()) + 1]),
                        axis=alt.Axis(grid=False, tickMinStep=1)),
                y=alt.Y("score:Q", title="Composite score (0-100)", scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(gridOpacity=0.25)),
                color=alt.Color("tier:N", title=None,
                                scale=alt.Scale(domain=list(TIER_COLORS), range=list(TIER_COLORS.values())),
                                legend=alt.Legend(orient="top")),
                tooltip=["rank", "where", "score", "tier", "hook", "cliffhanger", "judge_hook", "judge_cliff",
                         "judge_standalone", "spoiler", "audit_hook", "audit_cliff", "audit_spoiler", "flag"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch")
        with st.expander("Table view"):
            st.dataframe(df, width="stretch", hide_index=True)

    st.subheader("Ranked promos (post in this order)")
    f1, f2, f3 = st.columns(3)
    tiers = f1.multiselect("Tiers", list(TIER_COLORS), default=list(TIER_COLORS)[:2])
    hide_flag = f2.checkbox("Hide sexual/violence-flagged", value=False)
    topn = f3.number_input("Show top", 5, 200, 20)
    shown = [r for r in rows if r["rank"] is not None
             and (r.get("tier") or ("acquisition" if r["qualifies"] else "weak")) in tiers
             and (not hide_flag or r["judge"].get("content_flag", "none") in ("none", None))][: int(topn)]
    for r in shown:
        j = r["judge"]
        segs = "  ＋  ".join(f"E{s['episode']} {s['start']} → {s['end']}" for s in r["segments"])
        with st.container(border=True):
            h1, h2 = st.columns([4, 1])
            h1.markdown(f"#### #{r['rank']} · {segs}")
            t = r.get("tier") or ("acquisition" if r["qualifies"] else "weak")
            h2.markdown(f"**{r['duration_s']}s** · score **{r['score']}**  \n"
                        + {"acquisition": "✅ acquisition", "weak": "⚠️ weak"}.get(t, "🔁 re-engagement"))
            st.markdown(f"**Hook (0-3s):** {r['hook']}  \n**Cliffhanger:** {r['cliffhanger']}  \n"
                        f"**Why it works:** {r['why']}")
            if r.get("rank_reason"):
                st.markdown(f"**Why post it here:** {r['rank_reason']}")
            st.caption(f"Stranger judge ({r.get('judge_model') or 'n/a'}): hook {j.get('hook')} · cliffhanger "
                       f"{j.get('cliffhanger')} · standalone {j.get('standalone')} · spoiler risk "
                       f"{j.get('spoiler_risk')} · flag {j.get('content_flag')} — {j.get('verdict', '')}"
                       + (f"  ·  confusing: {j.get('confusing')}" if j.get("confusing") not in (None, "none") else "")
                       + (f"  ·  {r['repaired']}" if r.get("repaired") else ""))
            a = r.get("audit") or {}
            if a:
                st.caption(f"Show-aware auditor ({r.get('audit_model') or 'n/a'}): hook {a.get('hook')} · cliffhanger "
                           f"{a.get('cliffhanger')} · standalone {a.get('standalone')} · spoiler risk "
                           f"{a.get('spoiler_risk')} — {a.get('verdict', '')}"
                           + (f"  ·  suggested fix: {a.get('fix')}" if a.get("fix") not in (None, "", "none") else ""))
            with st.expander("Segments & clip script"):
                st.table(pd.DataFrame(r["segments"])[["episode", "start", "end", "duration_s", "blocks"]])
                st.code(r["clip_text"], language=None)

    st.subheader("Downloads")
    d1, d2, d3 = st.columns(3)
    d1.download_button("promos.json", json.dumps(rows, ensure_ascii=False, indent=2), "promos.json", "application/json")
    if st.session_state.get("csv"):
        d2.download_button("promos.csv", st.session_state["csv"], "promos.csv", "text/csv")
    if st.session_state.get("report"):
        d3.download_button("report.md", st.session_state["report"], "report.md", "text/markdown")
