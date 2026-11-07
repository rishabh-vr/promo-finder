# Promo Finder

Reads a microdrama's screenplays and finds the best **20–60 second promo clips**: a hook in the first
3 seconds, a cliffhanger at the cut, and understandable to someone who has never seen the show.
For every promo it gives the episode and start/end timestamp of each segment, the hook, the
cliffhanger, one line on why it works, a blind model QA score, and the order to post them.

Built for the Kuku AI Generalist assignment (Problem 1).

**Live app:** https://promo-finder.streamlit.app/ (paste one API key in the sidebar; the provider is detected from the key, so Gemini, OpenRouter, Groq and others all work. Or use rules-only mode without a key).

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
| Validate | code | The model names **block ids only** (including the exact hook block and cliffhanger block, which code uses to *measure* hook timing and how long the clip lingers after the cliffhanger); code turns them into timestamps, so it can't invent a time. Cuts are widened so no line or shot is cut in half; the 20–60s rule, segment overlap and cross-episode spans are enforced; clips slightly over 60s are trimmed from the start (never the cliffhanger end); near-duplicates are removed. |
| Judge | Blind LLM reviewer | Sees only the clip's script, no show context, like a stranger scrolling. Scores hook (first 3s), cliffhanger, standalone, spoiler risk (1–5) and flags sexual/violent content. The saved run used a different model from the proposer; model-family independence is not guaranteed by the code. |
| Rank | code (+ LLM text) | Deterministic and auditable: clips with judge hook ≥3 **and** cliffhanger ≥3 first (the brief's definition), then composite score (30% hook, 30% cliffhanger, 25% standalone, 15% low spoiler), max 2 per episode in the top 10. The LLM only writes the one-line "why post it here". |

## Free providers, fail-over and swapping models

Add any keys you have to `.env`; providers without a key are skipped. Then run `python doctor.py`
(or `python run.py --doctor`) to see what's usable.

| Provider | Free key | Used for (verified Nov 2026) |
|---|---|---|
| Google Gemini | https://aistudio.google.com/apikey | proposing (strongest); 20 req/day **per model**, so the chain spans several Gemini models |
| OpenRouter `:free` models | https://openrouter.ai/settings/keys | **auditing** with Nemotron 3 Ultra 550B when available; fallback for every role. The saved run's auditor used the same Gemini model family as its proposer. |
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
| Valid clips | 44 (2 of 46 candidate proposals rejected by code) | 50 |
| Qualify by blind judge (hook ≥3 and cliffhanger ≥3) | **70%** | 60% |
| Single-LLM-judged blind pairwise, same episode (43 pairs) | **23 wins** | 17 wins, 3 ties |
| Final tiers | 19 acquisition · 10 re-engagement · 15 weak | |

On spoiler risk, the blind judge and the show-aware auditor barely agree (Spearman 0.21; the blind
judge can't see twists), while cliffhanger scores agree well (0.72). That is why spoiler risk comes from
the auditor only.

The 70% qualification rate above is blind-judge-only (31/44). The final audited tiers contain 29
qualifying clips (19 acquisition + 10 re-engagement). Two candidate objects were rejected, with
three validation findings in total because one candidate had both an unknown block ID and a too-short
duration. The saved report was replayed from cache; its five-second runtime is not a fresh-run latency
measurement.
The pairwise preferences are from one LLM evaluator; they are preliminary model-based evidence, not
human-review or audience results.
**Statistical strength:** 23 wins to 17 is **not significant** (exact sign test p = 0.43; 95% Wilson
interval for the win rate is 0.42 to 0.72). At a 57% win rate about 175 decided pairs would be needed for
p < 0.05. The pairwise judge, the auditor and the generator were the **same model**
(`gemini-3-flash-preview`), so self-preference cannot be ruled out. `evaluate.py` now judges both A/B
orders and warns when the judge is the generator.

**Second pairwise check** (`submission/evaluation/pairwise_cross_family/`; judges from other model
families, mostly Nemotron (19 of 22 calls), with Qwen and gpt-oss; both A/B orders). The judge picked
whichever clip was shown first 58 times and the second only 24 times (p = 0.0002), and 17 of 43 pairs
flipped when the order was swapped. Counting only order-consistent verdicts gives LLM 14, rules 10,
ties 19 (p = 0.54): the same direction, and still not significant.

**On new, labelled shows** (3 synthetic shows, 24 planted promos and 24 resolved decoys, top 10 per show;
full table in `submission/evaluation/synthetic_benchmark/SUMMARY.md`):

| | LLM pipeline | Rule baseline |
|---|---|---|
| Planted promos found: episode-end / mid-episode | 10/12 · 7/12 | **12/12 · 9/12** |
| Planted promos found **with the hook inside 3 s** (of 24) | **9** | 3 |
| Clips sitting on a resolved-ending decoy (of 30) | **4** | 6 |

The honest reading: the rules find the planted moment at least as often (their episode-end bonus suits
plants at the episode's end). The pipeline's edge is where the clip *starts*. The sample is 24 moments
in 3 shows, so nothing here is statistically established. **Provenance:** every kept LLM clip records its
proposer in `gen_model`. Gemini's free quota had run out, so all of them came from the fallback,
**Nemotron**. Nemotron also co-wrote all three shows and audited the clips, so self-preference cannot
be excluded. Next step: more shows, with writer families disjoint from the pipeline's models.

The included raw `report.md` is intentionally preserved unedited as required by the assignment. Its
qualification label is from an older formatter: it calls 29/44 a stranger-judge result, while 29/44 is
the audited result and the blind-only result is 31/44. The report generator now labels both definitions.

## Testing on new shows (generalisation)

The interview runs the tool on a show it has never seen. Three kinds of test data are supported, and
each converts to Kuku's screenplay format, so the pipeline itself is unchanged:

| Source | Tool | Gold labels | What it tests |
|---|---|---|---|
| New LLM-written microdramas | `tools/synth_show.py` | planted promo (hook + cliffhanger) and a *resolved* decoy per episode; half the promos sit mid-episode | recall, hook timing, cut point, decoy avoidance on an unseen show |
| Film screenplays, e.g. the 15 TRIPOD test films | `tools/convert_screenplay.py` | TRIPOD's scene-level turning points (Papalampidi et al., 2019) | a different format and genre; does it find the story's key beats? |
| Any subtitles (.srt/.vtt), e.g. Blender's CC-BY *Tears of Steel* | `tools/convert_srt.py` | none | dialogue-only input with no camera or action lines |

```bash
python tools/synth_show.py --out ../test_data/synthetic/show_x --episodes 8           # new show + gold.json
python tools/convert_screenplay.py --script "Juno (film)_script.txt" --out ../test_data/tripod/juno        --tripod-gold TRIPOD_screenplays_test.csv --movie "Juno (film)"
python tools/convert_srt.py --in TOS-en.srt --out ../test_data/tears_of_steel
python tools/benchmark.py --shows ../test_data --out bench --mode baseline             # free
python tools/benchmark.py --shows ../test_data/synthetic --out bench_llm --mode llm    # uses keys
python tools/verify_claims.py --raw ../submission/raw_outputs --generator gemini-3-flash-preview
python evaluate.py --data <show> --out <run> --models groq:openai/gpt-oss-120b         # debiased A/B
```

`benchmark.py` reports recall and precision at K, **hook OK** (the clip starts within 3 s of the gold
hook), **cut OK** (the clip ends within -0.5 to +4 s of the gold cliffhanger), the decoy rate, and the
same metrics for random clips of equal length, with a Monte Carlo randomization p-value. If a clip cannot
fit in any episode, the matched-duration random comparison is marked unavailable instead of creating an
invalid null clip. `goldeval` sanitises gold
labels (for example, a cliffhanger tagged outside its own promo) and logs every fix in `label_warnings`.

**Evaluation hygiene built in**
- Pairwise A/B judges every pair in **both orders**. Only order-consistent verdicts count as wins, and the
  output reports the sign-test p-value, a Wilson 95% interval, the position bias, and whether the judge is
  the generator model.
- `verify_claims.py` recomputes every number quoted in the solution doc from the raw outputs (25/25
  reproduce) and checks every cut against the source screenplay.

## Cost and scale

The saved output is cached, so it does not provide a fresh-run latency or token-cost measurement.
Actual cost and elapsed time depend on provider, model, prompt size, quota, and retries. The free tier's
request limits are why prompts are batched; treat any cost estimate as provider-specific until a fresh
uncached run is recorded.

## Deploy the app (Streamlit Community Cloud, free): bring your own key

1. Push this repo to GitHub.
2. On share.streamlit.io choose **New app**, pick the repo, and set the main file to `app.py`.
3. **Do not add API keys to Secrets.** Each visitor pastes **their own** key into the single API-key field.
   The provider is auto-detected from the key prefix (`AIza…` Gemini, `sk-or-…` OpenRouter, `gsk_…` Groq,
   `csk-…` Cerebras, `hf_…` Hugging Face, `ghp_…`/`github_pat_…` GitHub) or picked from the dropdown (e.g. Mistral).

How keys are handled in the app (`app.py`, tested in `tests/test_keys.py`):
- The host's keys (`.env`, environment variables, Streamlit secrets) are **never used**. Only for
  self-hosting on your own machine can you opt in with `PROMO_ALLOW_SERVER_KEYS=1`.
- A pasted key stays in that visitor's session memory only. It is sent only to the chosen provider over
  HTTPS and is never written to disk, logs, outputs or the cache. It is redacted from error messages.
  **Forget key** clears it.
- Every visitor gets an isolated model state and response cache, so one person's bad key or spent quota
  can't block another, and nobody sees another visitor's results. Uploaded scripts are deleted after
  each run.
- Rules-only mode works with no key at all.

The CLI (`run.py`) still reads keys from your local `.env`, which is git-ignored and never committed.

## Repo map

- `promo/parse.py`: screenplay → episodes, scenes, timed blocks; transcript renderer
- `promo/clips.py`: block ids → timestamps, snapping, validation, deterministic checks
- `promo/prompts.py`: the three prompts (propose, judge, rank rationale)
- `promo/pipeline.py`: stages, scoring, ranking, exports, baseline experiment
- `promo/baseline.py`: rule-based proposer (control for the quality experiment; works without a key)
- `promo/llm.py`: provider-agnostic client with fallback chain, rate limit, cache
- `promo/stats.py`: Wilson CI, exact sign test, weighted kappa, bootstrap · `promo/goldeval.py`: gold-label scoring vs random
- `promo/kukufmt.py`: writer for Kuku-format episodes (used by every converter; round-trip tested)
- `tools/`: `synth_show.py`, `convert_screenplay.py`, `convert_srt.py`, `benchmark.py`, `verify_claims.py`
- `app.py`: Streamlit UI · `run.py`: CLI · `tests/`: unit tests on synthetic screenplays
