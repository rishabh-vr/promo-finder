"""Promo Finder: Streamlit front end.

Two ways in:
  1. Find promos: upload a show's episode screenplays (.txt files or one .zip), paste YOUR API key, run.
  2. Open results: upload a promos.json from an earlier run to browse it (no key needed).
No show data is bundled with the app (the screenplays are confidential).

Keys (bring your own):
  * One field. The provider is detected from the key's prefix (or chosen by hand).
  * The app never uses the host's keys (.env, environment, Streamlit secrets). Local self-hosting can
    opt in with PROMO_ALLOW_SERVER_KEYS=1.
  * A key lives only in this browser session's memory (st.session_state). It is sent only to that
    provider over HTTPS, never written to disk, logs, outputs or cache. "Forget key" clears it.
  * Each session gets its own model state and response cache, so one visitor's bad key, spent quota or
    uploaded scripts never affect or reach another visitor. Uploaded scripts are deleted after the run.
  * Everything rendered as HTML (model text, script text) is escaped.
"""
from __future__ import annotations

import html
import io
import json
import os
import shutil
import tempfile
import uuid
import zipfile

import altair as alt
import pandas as pd
import streamlit as st

from promo import pipeline
from promo.llm import LLM, PROVIDERS, LLMError, _State, load_dotenv, provider_key
from promo.parse import load_show

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide", initial_sidebar_state="auto")

# Tier colours: categorical slots 1-2 of the reference palette, dark-mode steps, validated on the
# app surface (#14151b): CVD ΔE 26.8, contrast >= 3:1. "weak" is a recessive neutral, not a hue.
TIER_COLORS = {"acquisition": "#3987e5", "re-engagement (spoils the plot)": "#d95926", "weak": "#6b6f7b"}
TIER_LABEL = {"acquisition": "Acquisition", "re-engagement (spoils the plot)": "Re-engagement", "weak": "Weak"}
E = html.escape

# ------------------------------------------------------------------ style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;1,600&family=JetBrains+Mono:wght@500&display=swap');
:root { --accent:#ff4d6d; --accent-2:#ff8a5c; --ink:#ecedf2; --ink-2:#a9acb8; --ink-3:#6f7383;
        --card:#16171f; --card-2:#1c1e28; --line:#262936; }
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea { font-family:'Inter',sans-serif; }
.block-container { padding-top:2.2rem; max-width:1180px; }
h1, h2, h3, h4 { letter-spacing:-0.01em; }
section[data-testid="stSidebar"] { background:#12131a; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top:1.4rem; }

/* hero */
.pf-hero { padding:6px 0 18px; }
.pf-eyebrow { font-size:.74rem; letter-spacing:.18em; text-transform:uppercase; color:var(--ink-3); font-weight:600; }
.pf-title { font-family:'Playfair Display',serif; font-size:3.1rem; line-height:1.05; margin:.25rem 0 .55rem;
            background:linear-gradient(92deg,#fff 0%,#ffd1d9 45%,var(--accent) 100%);
            -webkit-background-clip:text; background-clip:text; color:transparent; }
.pf-sub { color:var(--ink-2); font-size:1.02rem; max-width:720px; line-height:1.55; }
.pf-chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
.pf-chip { font-size:.78rem; color:var(--ink-2); border:1px solid var(--line); background:var(--card);
           padding:5px 11px; border-radius:999px; }
.pf-chip b { color:var(--ink); font-weight:600; }

/* metric cards */
div[data-testid="stMetric"] { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px 16px; }
div[data-testid="stMetricLabel"] p { color:var(--ink-3); font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; }
div[data-testid="stMetricValue"] { font-weight:700; }

/* clip cards */
.pf-card { background:linear-gradient(180deg,var(--card-2),var(--card)); border:1px solid var(--line);
           border-radius:16px; padding:18px 20px 14px; margin:2px 0 4px; }
.pf-head { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.pf-rank { font-family:'Playfair Display',serif; font-size:1.7rem; color:var(--ink); min-width:46px; }
.pf-ts { font-family:'JetBrains Mono',monospace; font-size:.8rem; color:var(--ink); background:#0f1015;
         border:1px solid var(--line); border-radius:8px; padding:4px 8px; }
.pf-plus { color:var(--ink-3); }
.pf-pill { font-size:.72rem; font-weight:600; letter-spacing:.04em; padding:4px 10px; border-radius:999px;
           border:1px solid; margin-left:auto; }
.pf-meta { color:var(--ink-3); font-size:.8rem; }
.pf-score { font-weight:700; color:var(--ink); }
.pf-body { display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-top:14px; }
@media (max-width: 800px) { .pf-body { grid-template-columns:1fr; } .pf-pill { margin-left:0; } }
.pf-k { font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-3); font-weight:600; margin-bottom:4px; }
.pf-v { color:var(--ink); font-size:.93rem; line-height:1.5; }
.pf-why { margin-top:12px; color:var(--ink-2); font-size:.9rem; line-height:1.5; border-left:2px solid var(--accent); padding-left:10px; }
.pf-bars { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:14px; }
@media (max-width: 800px) { .pf-bars { grid-template-columns:repeat(2,1fr); } }
.pf-bar-l { display:flex; justify-content:space-between; font-size:.74rem; color:var(--ink-3); margin-bottom:4px; }
.pf-bar-l b { color:var(--ink); font-weight:600; }
.pf-track { height:6px; background:#0f1015; border-radius:4px; overflow:hidden; }
.pf-fill { height:100%; border-radius:4px; background:linear-gradient(90deg,var(--accent),var(--accent-2)); }
.pf-warn { margin-top:10px; font-size:.82rem; color:#ffcf99; }

/* key box */
.pf-keyok { font-size:.8rem; color:#9fe0b8; margin-top:-4px; }
.pf-keyno { font-size:.8rem; color:var(--ink-3); margin-top:-4px; }
.pf-foot { color:var(--ink-3); font-size:.78rem; margin-top:28px; }
.stButton > button[kind="primary"] { background:linear-gradient(92deg,var(--accent),var(--accent-2)); border:0;
           font-weight:600; letter-spacing:.01em; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ per-session isolation
ALLOW_SERVER_KEYS = os.environ.get("PROMO_ALLOW_SERVER_KEYS") == "1"   # off by default: visitors bring keys
if "sid" not in st.session_state:
    st.session_state["sid"] = uuid.uuid4().hex
    st.session_state["llm_state"] = _State()          # this visitor's cool-downs / bad-key flags only
SESSION_CACHE = os.path.join(tempfile.gettempdir(), "promo_cache", st.session_state["sid"])
os.makedirs(SESSION_CACHE, exist_ok=True)

PROVIDER_NAMES = {"gemini": "Google Gemini", "openrouter": "OpenRouter", "groq": "Groq", "mistral": "Mistral",
                  "cerebras": "Cerebras", "github": "GitHub Models", "huggingface": "Hugging Face"}
PREFIXES = [("AIza", "gemini"), ("sk-or-", "openrouter"), ("gsk_", "groq"), ("csk-", "cerebras"),
            ("hf_", "huggingface"), ("github_pat_", "github"), ("ghp_", "github")]


def detect_provider(key: str) -> str | None:
    for prefix, prov in PREFIXES:
        if key.startswith(prefix):
            return prov
    return None


def make_llm(role, keys, chain=None):
    return LLM(role, chain=chain, keys=keys, cache_dir=SESSION_CACHE, use_env_keys=ALLOW_SERVER_KEYS,
               state=st.session_state["llm_state"])


def _redact(msg: str, keys: dict) -> str:
    """Belt and braces: never echo a key back, even if a provider includes it in an error."""
    for v in keys.values():
        if v and len(v) > 6:
            msg = msg.replace(v, v[:3] + "…redacted")
    return msg


# ------------------------------------------------------------------ sidebar: one key
with st.sidebar:
    st.markdown("### 🔑 API key")
    if ALLOW_SERVER_KEYS:
        load_dotenv()
    choice = st.selectbox("Provider", ["Auto-detect"] + [PROVIDER_NAMES[p] for p in PROVIDER_NAMES],
                          label_visibility="collapsed")
    api_key = st.text_input("API key", type="password", key="api_key", autocomplete="off",
                            placeholder="Paste your API key", label_visibility="collapsed").strip()
    keys: dict[str, str] = {}
    provider = None
    if api_key:
        provider = (detect_provider(api_key) if choice == "Auto-detect"
                    else next(p for p, n in PROVIDER_NAMES.items() if n == choice))
        if provider:
            keys[provider] = api_key
            st.markdown(f"<div class='pf-keyok'>● {E(PROVIDER_NAMES[provider])} key ready · this session only</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='pf-keyno'>Couldn't tell the provider from this key. Pick it above.</div>",
                        unsafe_allow_html=True)
    else:
        st.markdown("<div class='pf-keyno'>No key? Rules-only mode still works.</div>", unsafe_allow_html=True)
    if ALLOW_SERVER_KEYS:
        for p in PROVIDER_NAMES:
            if p not in keys and provider_key(p):
                keys[p] = provider_key(p)
    st.caption("Free keys: [Gemini](https://aistudio.google.com/apikey) · [OpenRouter](https://openrouter.ai/settings/keys)"
               " · [Groq](https://console.groq.com/keys)")
    if api_key and st.button("Forget key", width="stretch"):
        st.session_state.pop("api_key", None)
        st.session_state["llm_state"] = _State()
        shutil.rmtree(SESSION_CACHE, ignore_errors=True)
        st.rerun()
    st.caption("Your key stays in this browser session's memory, goes only to that provider, and is never saved.")

    roles = {r: make_llm(r, keys) for r in ("propose", "judge", "audit")}
    with st.expander("Advanced"):
        use_audit = st.toggle("Show-aware audit", value=True,
                              help="A second opinion that knows the whole story: catches spoilers the blind judge can't.")
        chunk = st.slider("Episodes per request", 1, 10, pipeline.CHUNK,
                          help="Bigger = fewer requests and more cross-episode context. Split automatically if too big.")
        gen_models = st.text_area("Generator models", ",\n".join(roles["propose"].entries), height=90)
        judge_models = st.text_area("Blind-judge models", ",\n".join(roles["judge"].entries), height=90)
        audit_models = st.text_area("Auditor models", ",\n".join(roles["audit"].entries), height=80)
        st.caption("Models available for your key are listed in fail-over order. Edit to reorder or remove.")

# ------------------------------------------------------------------ hero
st.markdown("""
<div class="pf-hero">
  <div class="pf-eyebrow">Microdrama · Promo discovery</div>
  <div class="pf-title">Promo Finder</div>
  <div class="pf-sub">Drop in a show's screenplays and get the 20–60 second clips worth posting: a hook in the first
  three seconds, a cut on a cliffhanger, and enough context for a stranger to follow, ranked in the order to post.</div>
  <div class="pf-chips">
    <span class="pf-chip"><b>Exact timestamps</b> computed by code, never invented</span>
    <span class="pf-chip"><b>Blind judge</b> scores each clip as a stranger</span>
    <span class="pf-chip"><b>Spoiler-aware</b> ranking</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_run, tab_view = st.tabs(["Find promos", "Open results"])


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
    files = st.file_uploader("Episode screenplays: .txt files, or one .zip of them", type=["txt", "zip"],
                             accept_multiple_files=True)
    c1, c2, c3 = st.columns([2, 1.2, 1])
    modes = ["AI pipeline", "Rules only (no key)"]
    mode = c1.segmented_control("Mode", modes, default=modes[0] if keys else modes[1], label_visibility="collapsed") or modes[1]
    ep_filter = c2.text_input("Episodes", placeholder="Episodes, e.g. 1-10 (optional)", label_visibility="collapsed")
    go = c3.button("Find promos", type="primary", disabled=not files, width="stretch")
    if mode == modes[0] and not keys:
        st.info("Paste your API key in the sidebar to use the AI pipeline, or switch to rules only.")
    if go:
        data_dir = _extract(files)
        try:
            eps = load_show(data_dir)
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(data_dir, ignore_errors=True)
            st.error(f"Could not parse screenplays: {e}")
            st.stop()
        subset = None
        if ep_filter.strip():
            from run import parse_eps
            subset = parse_eps(ep_filter.strip())
        out_dir = tempfile.mkdtemp(prefix="out_")
        pipeline.CHUNK = chunk
        llm = judge = auditor = None
        if mode == modes[0]:
            split = lambda s: [x.strip() for x in s.replace("\n", ",").split(",") if x.strip()]
            llm = make_llm("propose", keys, split(gen_models))
            judge = make_llm("judge", keys, split(judge_models))
            if use_audit:
                auditor = make_llm("audit", keys, split(audit_models))
            if not llm.entries or not judge.entries:
                shutil.rmtree(data_dir, ignore_errors=True)
                st.error("No usable model for this key. Check the key, or switch to rules only.")
                st.stop()
        with st.status(f"Reading {len(eps)} episodes · {sum(e.duration for e in eps) / 60:.0f} min of show",
                       expanded=True) as status:
            try:
                rows = pipeline.run(data_dir, out_dir, "llm" if llm else "baseline", llm, judge, subset,
                                    cb=lambda m: st.write(m), eps=eps,
                                    show_name=os.path.splitext(files[0].name)[0] if len(files) == 1 else None,
                                    audit_llm=auditor)
                status.update(label=f"Found {len(rows)} clips", state="complete", expanded=False)
            except LLMError as e:
                status.update(label="Model error", state="error")
                st.error(_redact(str(e), keys))
                shutil.rmtree(data_dir, ignore_errors=True)
                shutil.rmtree(out_dir, ignore_errors=True)
                st.stop()
        st.session_state["rows"] = rows
        st.session_state["stats"] = json.load(open(os.path.join(out_dir, "run_stats.json"), encoding="utf-8"))
        st.session_state["report"] = open(os.path.join(out_dir, "report.md"), encoding="utf-8").read()
        st.session_state["csv"] = open(os.path.join(out_dir, "promos.csv"), encoding="utf-8-sig").read()
        # the uploaded scripts are confidential: keep results in this session's memory, delete the files
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)

with tab_view:
    up = st.file_uploader("promos.json from an earlier run (optionally run_stats.json too)", type=["json"],
                          accept_multiple_files=True, key="view")
    for f in up or []:
        blob = json.loads(f.getvalue())
        if isinstance(blob, list):
            st.session_state["rows"] = blob
        elif isinstance(blob, dict) and "summary" in blob:
            st.session_state["stats"] = blob


# ------------------------------------------------------------------ results
def tier_of(r):
    return r.get("tier") or ("acquisition" if r.get("qualifies") else "weak")


def bar(label, v, risk=False):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return f"<div><div class='pf-bar-l'><span>{E(label)}</span><b>–</b></div><div class='pf-track'></div></div>"
    pct = max(0.0, min(1.0, (x - 1) / 4)) * 100
    fill = "background:#6b6f7b" if risk else ""      # risk is not "good": neutral fill, not the accent
    return (f"<div><div class='pf-bar-l'><span>{E(label)}</span><b>{x:g}<span style='color:#6f7383'>/5</span></b></div>"
            f"<div class='pf-track'><div class='pf-fill' style='width:{pct:.0f}%;{fill}'></div></div></div>")


def card(r) -> str:
    t = tier_of(r)
    col = TIER_COLORS.get(t, "#6b6f7b")
    segs = "<span class='pf-plus'>+</span>".join(
        f"<span class='pf-ts'>E{int(s['episode'])} {E(str(s['start']))} → {E(str(s['end']))}</span>" for s in r["segments"])
    j, a = r.get("judge") or {}, r.get("audit") or {}
    fin = r.get("final") or {}
    pick = lambda k: fin.get(k, a.get(k, j.get(k)))
    warns = []
    ck = r.get("checks") or {}
    if ck.get("hook_event_within_3s") is False:
        warns.append("Hook lands after 3 s: consider starting later.")
    if ck.get("stitch_has_shared_speaker") is False:
        warns.append("Stitched segments share no speaking character: check the cut is clear.")
    if j.get("content_flag") not in (None, "none"):
        warns.append(f"Content flag: {j.get('content_flag')}. Check platform rules before posting.")
    why = r.get("rank_reason") or r.get("why") or ""
    return f"""
<div class="pf-card">
  <div class="pf-head">
    <span class="pf-rank">#{E(str(r.get('rank')))}</span>{segs}
    <span class="pf-meta">{E(str(r.get('duration_s')))}s · score <span class="pf-score">{E(str(r.get('score')))}</span></span>
    <span class="pf-pill" style="color:{col};border-color:{col}55;background:{col}1a">{E(TIER_LABEL.get(t, t))}</span>
  </div>
  <div class="pf-body">
    <div><div class="pf-k">Hook · first 3 s</div><div class="pf-v">{E(str(r.get('hook') or ''))}</div></div>
    <div><div class="pf-k">Cliffhanger · at the cut</div><div class="pf-v">{E(str(r.get('cliffhanger') or ''))}</div></div>
  </div>
  <div class="pf-why">{E(str(why))}</div>
  <div class="pf-bars">{bar('Hook', pick('hook'))}{bar('Cliffhanger', pick('cliffhanger'))}{bar('Standalone', pick('standalone'))}{bar('Spoiler risk', pick('spoiler_risk'), risk=True)}</div>
  {''.join(f"<div class='pf-warn'>⚠ {E(w)}</div>" for w in warns)}
</div>"""


rows = st.session_state.get("rows")
if rows:
    stats = st.session_state.get("stats") or {}
    summ = stats.get("summary") or pipeline.summarize(rows)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    m = st.columns(4)
    m[0].metric("Clips found", summ["clips"])
    m[1].metric("Ready for new viewers", (summ.get("by_tier") or {}).get("acquisition", summ["qualifying"]),
                help="Hook and cliffhanger ≥3 of 5, hook within 3 s, and low spoiler risk")
    m[2].metric("Episodes covered", summ.get("episodes_with_qualifying", "–"),
                help="Episodes with at least one qualifying clip")
    m[3].metric("Rejected by code checks", stats.get("n_rejected", "–"))

    df = pd.DataFrame([{
        "rank": r["rank"], "episode": r["episode"], "score": r["score"], "tier": TIER_LABEL.get(tier_of(r), tier_of(r)),
        "where": " + ".join(f"E{s['episode']} {s['start']}–{s['end']}" for s in r["segments"]),
        "duration_s": r["duration_s"], "hook": r["hook"], "cliffhanger": r["cliffhanger"],
        "judge_hook": (r.get("judge") or {}).get("hook"), "judge_cliff": (r.get("judge") or {}).get("cliffhanger"),
        "spoiler": (r.get("audit") or r.get("judge") or {}).get("spoiler_risk"),
    } for r in rows])

    chart_df = df.dropna(subset=["score"])
    if not chart_df.empty:
        st.markdown("#### Where the promos are")
        chart = (
            alt.Chart(chart_df)
            .mark_circle(size=140, opacity=0.95, stroke="#0f1015", strokeWidth=2)
            .encode(
                x=alt.X("episode:Q", title="Episode", scale=alt.Scale(domain=[0, int(df.episode.max()) + 1]),
                        axis=alt.Axis(grid=False, tickMinStep=1)),
                y=alt.Y("score:Q", title="Score (0–100)", scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(gridOpacity=0.12)),
                color=alt.Color("tier:N", title=None,
                                scale=alt.Scale(domain=[TIER_LABEL[t] for t in TIER_COLORS], range=list(TIER_COLORS.values())),
                                legend=alt.Legend(orient="top", labelColor="#a9acb8")),
                tooltip=["rank", "where", "score", "tier", "hook", "cliffhanger", "judge_hook", "judge_cliff", "spoiler"],
            )
            .properties(height=250, background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor="#a9acb8", titleColor="#6f7383", domainColor="#262936", tickColor="#262936",
                            gridColor="#a9acb8")
        )
        st.altair_chart(chart, width="stretch")
        with st.expander("Table view"):
            st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("#### Ranked promos · post in this order")
    f1, f2, f3 = st.columns([2.2, 1.2, 1])
    tiers = f1.pills("Tiers", list(TIER_COLORS), default=list(TIER_COLORS)[:2], selection_mode="multi",
                     format_func=lambda t: TIER_LABEL[t], label_visibility="collapsed") or []
    hide_flag = f2.toggle("Hide flagged content", value=False)
    topn = f3.number_input("Show top", 5, 200, 20, label_visibility="collapsed")
    shown = [r for r in rows if r.get("rank") is not None and tier_of(r) in tiers
             and (not hide_flag or (r.get("judge") or {}).get("content_flag", "none") in ("none", None))][: int(topn)]
    if not shown:
        st.caption("No clips match these filters.")
    for r in shown:
        st.markdown(card(r), unsafe_allow_html=True)
        with st.expander("Details · scores, segments and clip script"):
            j, a = r.get("judge") or {}, r.get("audit") or {}
            st.caption(f"Blind judge ({r.get('judge_model') or 'n/a'}): {j.get('verdict', '')}")
            if a:
                st.caption(f"Show-aware auditor ({r.get('audit_model') or 'n/a'}): {a.get('verdict', '')}"
                           + (f" · suggested fix: {a.get('fix')}" if a.get("fix") not in (None, "", "none") else ""))
            ck = r.get("checks") or {}
            if ck.get("hook_offset_s") is not None:
                st.caption(f"Code-measured hook lands at {ck['hook_offset_s']:.1f}s"
                           + (f" · clip runs {ck['cliff_tail_s']:.1f}s past the cliffhanger"
                              if ck.get("cliff_tail_s") is not None else ""))
            st.dataframe(pd.DataFrame(r["segments"])[["episode", "start", "end", "duration_s", "blocks"]],
                         hide_index=True, width="stretch")
            st.code(r.get("clip_text", ""), language=None)

    st.markdown("#### Download")
    d1, d2, d3 = st.columns(3)
    d1.download_button("promos.json", json.dumps(rows, ensure_ascii=False, indent=2), "promos.json",
                       "application/json", width="stretch")
    if st.session_state.get("csv"):
        d2.download_button("promos.csv", st.session_state["csv"], "promos.csv", "text/csv", width="stretch")
    if st.session_state.get("report"):
        d3.download_button("report.md", st.session_state["report"], "report.md", "text/markdown", width="stretch")

st.markdown("<div class='pf-foot'>The model only points at script blocks; code computes every timestamp and never "
            "cuts a line in half. Scores are model judgements: an editor approves before anything is posted.</div>",
            unsafe_allow_html=True)
