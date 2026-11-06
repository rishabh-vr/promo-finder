# Promo Finder

Reads a microdrama's screenplays and finds the best **20–60 second promo clips**: a hook in the first
3 seconds, a cliffhanger at the cut, and understandable to someone who has never seen the show.
For every promo it gives the episode and start/end timestamp of each segment, the hook, the
cliffhanger, one line on why it works, an independent QA score, and the order to post them.

Built for the Kuku AI Generalist assignment (Problem 1).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env                 # paste a free Gemini key: https://aistudio.google.com/apikey

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

## Swap the model

Every call goes to an OpenAI-compatible `/chat/completions` endpoint. Change models with env vars, no
code change:

```bash
LLM_PRESET=gemini|groq|openrouter|anthropic|openai     # or LLM_BASE_URL=...
LLM_PROPOSE_MODELS=gemini-3-flash-preview,gemini-3.8-flash   # fallback chain, first = preferred
LLM_JUDGE_MODELS=gemini-3.1-flash-lite,gemini-flash-lite-latest
```

When a model's daily quota is spent or it is overloaded, the client moves to the next model in the
chain. Every response is cached on disk, so an interrupted run resumes for free.

## Cost and scale

A 50-episode show (~105k tokens of script) takes ~10 proposal calls + ~10 judge calls + 1 ranking
call ≈ **300k input / 40k output tokens**. At paid flash-tier prices that's roughly **$0.05–0.15 per
show**, and a few minutes per show. Shows are independent, so many can run in parallel. The free tier
(20 requests/day/model) is the only reason the prompts are batched so heavily.

## Repo map

- `promo/parse.py`: screenplay → episodes, scenes, timed blocks; transcript renderer
- `promo/clips.py`: block ids → timestamps, snapping, validation, deterministic checks
- `promo/prompts.py`: the three prompts (propose, judge, rank rationale)
- `promo/pipeline.py`: stages, scoring, ranking, exports, baseline experiment
- `promo/baseline.py`: rule-based proposer (control for the quality experiment; works without a key)
- `promo/llm.py`: provider-agnostic client with fallback chain, rate limit, cache
- `app.py`: Streamlit UI · `run.py`: CLI · `tests/`: unit tests on synthetic screenplays
