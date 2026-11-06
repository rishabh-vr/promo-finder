# Promo Finder

Reads a microdrama's screenplays and finds the best **20–60 second promo clips**: a hook in the first
3 seconds, a cliffhanger at the cut, and understandable to someone who has never seen the show.
For every promo it gives the episode and start/end timestamp of each segment, the hook, the
cliffhanger, one line on why it works, an independent QA score, and the order to post them.

Built for the Kuku AI Generalist assignment (Problem 1).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env                 # paste any free keys you have (links inside)
python doctor.py                     # which providers / models are usable right now

python run.py --data "path/to/show folder" --out out                 # full pipeline
python run.py --data "path/to/show folder" --out out --compare-baseline 50   # + quality experiment
python run.py --data "path/to/show folder" --out out_rules --mode baseline   # no key needed
streamlit run app.py                                                 # UI: upload scripts, browse results
python -m pytest -q tests                                            # unit tests (synthetic data)
```

`--data` is a folder of `episode-*.txt` screenplays in Kuku's format. The show itself is **not** in
this repo (confidential).

**Outputs** (`--out`): `report.md` (shortlist to read), `promos.json` / `promos.csv` (every valid clip:
segments, timestamps, scores, clip script), `rejected.json` (every proposal the code threw out, and
why), `run_stats.json` (calls, tokens, models used, failovers, summary metrics).

## How it works

```
screenplays ─► 1 PARSE (code) ─► 2 PROPOSE (LLM) ─► 3 VALIDATE (code) ─► 4 JUDGE (2nd LLM, blind) ─► 5 RANK (code) ─► report
                timed blocks      block ids only     timestamps, 20-60s,    stranger's view: hook,      qualifiers first,
                                  5 episodes/call    no half-cut lines,     cliffhanger, standalone,    composite score,
                                  + story bible      repair, de-dupe        spoiler, content flag       ≤2/episode in top 10
```

| Stage | Who | Why that split |
|---|---|---|
| Parse | code | The format is regular; a parser is exact and free. |
| Propose | LLM | Judging drama needs language understanding. The model sees a one-line **story bible** of every episode plus full transcripts of 5 consecutive episodes, so it knows the plot and can **stitch moments across episodes**. |
| Validate | code | The model names **block ids only**; code turns them into timestamps, so it can't invent a time. Cuts are widened so no line or shot is cut in half; the 20–60s rule, segment overlap and cross-episode spans are enforced; clips slightly over 60s are trimmed from the start (never the cliffhanger end); near-duplicates are removed. |
| Judge | a **different** LLM | Sees only the clip's script, no show context, like a stranger scrolling. Scores hook (first 3s), cliffhanger, standalone, spoiler risk (1–5) and flags sexual/violent content. A different model reduces self-preference. |
| Rank | code (+ LLM text) | Deterministic and auditable: clips with judge hook ≥3 **and** cliffhanger ≥3 first (the brief's definition), then composite score (30% hook, 30% cliffhanger, 25% standalone, 15% low spoiler), max 2 per episode in the top 10. The LLM only writes the one-line "why post it here". |

## Free providers, fail-over and swapping models

Add any keys you have to `.env`; providers without a key are skipped. Then run `python doctor.py`
(or `python run.py --doctor`) to see what's usable.

| Provider | Free key | Used for (verified Nov 2026) |
|---|---|---|
| Google Gemini | https://aistudio.google.com/apikey | proposing (strongest); 20 req/day **per model**, so the chain spans several Gemini models |
| OpenRouter `:free` models | https://openrouter.ai/settings/keys | **auditing** with Nemotron 3 Ultra 550B (a different model family = independent opinion); back-up for every role |
| Groq | https://console.groq.com/keys | **judging** (gpt-oss-120b, Qwen 3.8; ~1-3 s per call; small per-minute token limits, so big batches auto-split) |
| Hugging Face | https://huggingface.co/settings/tokens | last-resort back-up (tiny monthly credit; use a fine-grained "Inference Providers" token) |
| Ollama (local) | https://ollama.com/download | unlimited offline fall-back (`OLLAMA_ENABLED=1`) |
| Mistral, Cerebras, OpenAI, Anthropic | paid | supported if you have a key (`MISTRAL_API_KEY`, etc.) |

GitHub Models is not in the chains: its endpoint answered `200 OK` with no completion during testing.

**How fail-over keeps runs fast:**
- Each role (propose / judge / audit) has an ordered chain of `provider:model` entries across all
  providers, with defaults in `promo/llm.py`. Override with `LLM_PROPOSE_CHAIN`, `LLM_JUDGE_CHAIN`
  and `LLM_AUDIT_CHAIN`.
- **Daily quota spent:** the client switches instantly and remembers it on disk until the reset, so
  later runs skip that model without trying.
- **Per-minute limit or overload:** the model gets a short cool-down while traffic flows to the next one.
- **Prompt too big for a small tier:** learned per model. If nothing can fit, the batch is automatically
  split into smaller requests.
- **Missing models:** dropped up front using each provider's model catalogue.
- **Speed:** requests run in parallel (`PROMO_WORKERS=4`), each model has its own rate limit, and a
  model whose next slot is more than `LLM_PATIENCE` seconds away is passed over for a ready one.
- **Caching:** every response is cached on disk, so re-runs and interrupted runs don't pay twice.

Any other OpenAI-compatible endpoint works via `LLM_BASE_URL` + `LLM_API_KEY`.

## Results on a 50-episode show (Kuku input pack)

| | LLM pipeline | Rule baseline |
|---|---|---|
| Valid clips | 44 (2 of 46 proposals rejected by code) | 50 |
| Qualify by blind judge (hook ≥3 and cliffhanger ≥3) | **70%** | 60% |
| Blind head-to-head, same episode (43 pairs) | **23 wins** | 17 wins, 3 ties |
| Final tiers | 19 acquisition · 10 re-engagement · 15 weak | |

On spoiler risk, the blind judge and the show-aware auditor barely agree (Spearman 0.21; the blind
judge can't see twists), while cliffhanger scores agree well (0.72). That is why spoiler risk comes from
the auditor only.

## Cost and scale

A 50-episode show (~105k tokens of script) took ~20 calls (10 proposal, 4 judge, 2 audit, 1 rationale,
plus retries), ≈ **0.4M input / 20k output tokens**: cents per show at flash-tier prices, 9 minutes on
the rate-limited free tier. Proposal calls are independent and could run concurrently on a paid tier.
The free tier (20 requests/day/model) is why prompts are batched so heavily.

## Deploy the app (Streamlit Community Cloud, free)

1. Push this repo to GitHub.
2. On share.streamlit.io choose **New app**, pick the repo, and set the main file to `app.py`.
3. Optional: under **Advanced settings → Secrets**, add any of `GEMINI_API_KEY`, `MISTRAL_API_KEY`,
   `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `GITHUB_MODELS_TOKEN` and `HF_TOKEN` (TOML,
   e.g. `GEMINI_API_KEY = "..."`). Visitors can then run without their own keys, and the keys are never
   sent to the browser. Without them, visitors paste keys in the sidebar or use the rules-only mode.

## Repo map

- `promo/parse.py`: screenplay → episodes, scenes, timed blocks; transcript renderer
- `promo/clips.py`: block ids → timestamps, snapping, validation, deterministic checks
- `promo/prompts.py`: the three prompts (propose, judge, rank rationale)
- `promo/pipeline.py`: stages, scoring, ranking, exports, baseline experiment
- `promo/baseline.py`: rule-based proposer (control for the quality experiment; works without a key)
- `promo/llm.py`: provider-agnostic client with fallback chain, rate limit, cache
- `app.py`: Streamlit UI · `run.py`: CLI · `tests/`: unit tests on synthetic screenplays
