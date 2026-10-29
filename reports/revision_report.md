# Revision report: pipeline for FAccT 2027

Final report of the code revision described in `REVISION_PLAN.md` (Phases 0–8), written on 2026-10-29. Every value confirmed by the author is logged in [`reports/verified_values.md`](verified_values.md).

## Status

- **Code:** all phases are implemented and committed. The pre-revision code is tagged `pre-facct-revision`.
- **Tests:** 296 pytest tests pass on CPU with tiny random models (`pytest`, about 30 s).
- **Dry run:** `python run_pipeline_dry.py` runs the whole pipeline on tiny models (CPU, no network) and passes: sanity, Logit Lens, behavioral generation, Tuned Lens training and re-measurement, analysis and statistics, 40 figures, both LaTeX tables, integrity checks, validation agreement. About 105 s, about 1.1 GB RAM.
- **No real-model runs**, no downloads of weights, and no Tuned Lens training were done. Everything that needs a GPU is a command in `RUNNING_1GPU.md` and `RUNNING_multiGPU.md`.
- **Runs are blocked** until the values below are confirmed. The scripts refuse to start, with a message listing what is pending.

## Blocking the runs (unconfirmed)

Updated 2026-10-29, after the author's decisions: Russian left the study (five languages: PT, EN, DE, IT, FR); the lightbulb fact was removed (11 facts); the facts are confirmed; the FR, DE and IT prompts were reviewed by native speakers. Everything still proposed is listed, with an ID per item, in [`reports/review_pt_en_and_support.md`](review_pt_en_and_support.md).

| # | What | Where | Blocks |
| --- | --- | --- | --- |
| 1 | PT and EN prompts other than the 6 PT prompts confirmed in Phase 0 (318 prompts; the code doesn't gate on them, but they aren't confirmed) | `config/prompts.yaml`; section a of the review document | should be confirmed before the lens and behavioral runs |
| 2 | Neutral baseline prompts: `status: proposed`; FR, DE and IT blocks still marked for native review (the native review covered `config/prompts.yaml`) | `config/baseline_prompts.yaml`; section b | lens runs |
| 3 | Article-confound forcing words: `status: proposed` | `config/forcing.yaml`; section c | lens runs |
| 4 | Behavioral surface forms: `status: proposed` | `config/behavioral.yaml`; section d | behavioral analysis (generation itself can run) |

**Tuned Lens training can start now:** it needs only the confirmed facts. `python train_tuned_lens.py --model llama` (then `mistral`, `qwen`, `gemma`).

**Figure width:** set to 5.95 in (430.00462pt, acmart v2.21 manuscript), measured by the author.

**Tokenization check** (2026-10-29, 11 tokenizers; [`reports/tokenization_summary.md`](tokenization_summary.md)):
- **Resolved by prefix forcing** (shared token forced, P(prefix) × P(next | prefix)):
  - radio places in PT and IT (São/San Paulo vs Petersburgo/Pietroburgo), all models;
  - cinema artifacts in PT and IT with the Mistral tokenizers (`cin` → `emat` / `et`);
  - Mendeleev/Meyer in QA (`M` → `end` / `ey`) for Llama, Mistral and Qwen instruct.
- **Still dropped:** no word. **Indicator cells skipped:** none.

## What changed, by phase

| Phase | Commit | Changes |
| --- | --- | --- |
| 0, fixes 1–4 | d494414 | Final norm applied once (the last hidden state is already normalized); Gemma 2 logit softcapping; lens sanity check on every forward (last layer must reproduce the model's logits); bf16 by default. |
| 0, fixes 5–6 | c15b87e | Exact probabilities over token IDs instead of substring matching on the top-10 (" Bell" no longer catches " bella"); word-boundary, accent-insensitive behavioral matching; one centered smoothing that shrinks at the edges instead of zero-padding. |
| 0, fixes 7–11 | 44a4cbf | Accented prompts; expected counts derived from config; `.gitignore`; one metrics implementation (`analyze.py`) and one figures implementation (`make_figures.py`); duplicate scripts and the `visualization/` package removed. |
| 1 | 40d84c3 | Facts, candidates, stakes, single-name sets (Russian by case), narrative indicators and 1,116 prompts (3 paraphrases per type) in YAML, with validation; contraction flag (`article_blocked`); tokenization check in `verify_setup.py`; run guard on unconfirmed values. |
| 2, part 1 | 6349397 | Pluggable lens interface; exact per-candidate probability over distinct token IDs; neutral baseline (108 prompts mirroring each real paraphrase); KL(lens ‖ final) per layer; top-20 for inspection. |
| 2, part 2 | 3370154 | `run_lens.py` with checkpointed Parquet tables and run manifest; CS_e, baseline-adjusted CS, k*, CS*, suppression, decision-zone mean, ΔCS; article-confound test (article + plural noun, contractions, shared prefixes); indicator metrics; completion and QA formats; old JSON pipeline removed. |
| 3 | 8b9d19e | Model registry (11 models pinned to commits); output head located by architecture, with shape checks; NF4 for Gemma 2 27B; formats by base/instruct; per-model sanity script; GPU-hours in the manifest. |
| 4 | f40c12c | Own Tuned Lens (same recipe as the tuned-lens package); multilingual corpus with held-out split and fact filter; resumable training (resuming gives the same lens as an uninterrupted run); held-out KL evaluation; `--lens tuned`; agreement analysis with the Logit Lens. |
| 5 | e5035fb | Behavioral generation with explicit decoding settings; candidate and indicator extraction (Latin + Cyrillic, all Russian cases, exclusions for common words); rates per cell; empirical English dominant; report questions (Reis before Meucci in German, Santos-Dumont in French); stratified validation sample and agreement (Cohen's κ). |
| 6 | 621f735 | Stratified bootstrap CIs; stake vs non-stake permutation tests (Holm); two mixed models with crossed random effects for fact and model; internal ↔ behavioral join. |
| 7 | 7d9e9c0 | Figures 1–6 + appendix heatmaps as PDF + PNG (then two-column ACM sizes; now single-column, see below), Okabe-Ito palette, main and appendix versions; LaTeX results tables (compact and full). |
| 8 | 2d4de97 | `--model` / `--gpus` / `--dtype` / `--sanity-only` entry points; precision and commit checked on resume; run queue for several GPUs; dry run; pinned requirements; README; anonymized copy. |
| Finish | 50f8571 | Lock against two processes training the same Tuned Lens; bf16-translator test; corrected training-memory estimate; RUNNING guides updated (not committed, as agreed); this report. |

| Phase 1 answers | 7693874 | Facts confirmed and logged (33 entries); Russian forms pending review tracked per candidate; place/artifact prompts only with two candidates (54 artifact prompts removed: telephone, radio, calculus; skipped per tokenizer when needed, recorded); tokenization check on the 11 real tokenizers; figures sized for the single-column acmart manuscript (width pending measurement); behavioral report no longer fails when a candidate is never cited. |

| Later decisions | this commit | Russian and the lightbulb fact removed (5 languages, 11 facts, 810 prompts); FR/DE/IT prompts reviewed; shared first tokens resolved by prefix forcing in every measurement, each case logged (`prefix_forcing` table); figure width 5.95 in; `reports/review_pt_en_and_support.md`. |

## Decisions (asked and answered)

| Phase | Question | Option chosen |
| --- | --- | --- |
| 0 | Smoothing for curve figures | Two versions: 3-layer window and raw |
| 0 | Unused `visualization/` package | Delete it |
| 0 | Which old figures to port | Fig. 3 only |
| 0 | Duplicate scripts | Delete all but `monitor.py` |
| 0 | Entity sets | Leave for Phase 1 |
| 0 | PT/IT prompt accents | Correct as shown |
| 1 | Cinema dominant | Lumière brothers |
| 1 | Photography | Daguerre dominant, Niépce a separate candidate |
| 1 | Stakes | Nationality stake for every candidate (place-of-event only Santos-Dumont FR, as in the plan) |
| 1 | Prepositions that block plurals (PT/IT) | Keep paraphrase 1, fix paraphrases 2–3, flag |
| 1 | Russian grammatical case | Case-aware slots |
| 1 | "Common word" threshold | wordfreq Zipf ≥ 4.0 |
| 1 | Control facts | F1–F3 + QA |
| 1 | Verbs | Natural verb per fact (invented / developed / discovered) |
| 2 | Neutral baseline | Match each prompt type (108 prompts) |
| 2 | Baseline comparison | Difference P(e) − B_e |
| 2 | Article verdict | Against the candidate's own peak |
| 2 | ΔCS pairing | Mean over paraphrases |
| 2 | Old figures | Port minimally to the new tables |
| 3 | Model IDs, licenses, gating | Confirmed as shown |
| 3 | Sabiá base model | huggyllama/llama-7b |
| 3 | Revisions | Pin each model to a commit |
| 3 | Gemma 2 27B quantization | NF4, bf16 compute, no double quantization |
| 4 | Tuned Lens implementation | Own implementation for all four models |
| 4 | Corpus | FineWeb-2 + FineWeb |
| 4 | Fact documents in the corpus | Filter them out |
| 4 | Budget | 16M tokens per model |
| 5 | Sampling | Pure temperature 0.7 (models' own decoding defaults overridden) |
| 5 | Surface forms | New reviewed list |
| 5 | Empirical dominant | Pooled English answers |
| 5 | Validation sample | Stratified, ~400 |
| 6 | Main metric | Decision-zone mean |
| 6 | Bootstrap | Stratified over F1–F3 |
| 6 | Stake test | Include dominant candidates with the mirrored score |
| 6 | Mixed model | Two fits |
| 7 | Main-text models | Four instruct models |
| 7 | Fig. 2 | Per-model points with CIs |
| 7 | Results table | One row per alternative; compact = stake languages + EN |
| 7 | Fig. 4 | Both panels (CS* and decision-zone CS) |
| — | Tracked `.env` | Leave as is |
| Phase 1 answers | All "To check" items and Phase 1 questions | As logged in `reports/verified_values.md` (2026-10-29) |
| Phase 1 answers | Indicator prompts | Only with at least two candidates with valid tokens, skipped per fact × language otherwise |
| Phase 1 answers | Paper format | acmart manuscript, single column; every figure at `\textwidth` |
| Later | Languages | PT, EN, DE, IT, FR (Russian removed); lightbulb fact removed |
| Later | Native review | FR, DE, IT prompts confirmed by native speakers |
| Later | Shared first tokens | Force the prefix and combine P(prefix) × P(next \| prefix), any pair, any format; log every case |
| Later | Figure width | 5.95 in (430.00462pt) |

## Choices I made (implementation details worth knowing)

- **Normalized depth** is k/L in the new tables, as in the plan. The old code used (k−1)/(L−1).
- **Contraction flag** is named `article_blocked`, not `plural_blocked`, because it also covers places that take an article.
- **Neutral prompt topics:** *the novel / its author* for names, *the bridge* for places, *the first ship* for artifacts.
- **Article test, second step:** after a forced article, a plural noun ("os" → "irmãos") is forced too. Otherwise "os irmãos Wright" would be scored as suppression.
- **CS left undefined (NaN)** when either candidate has no token IDs, e.g. when the dominant candidate has no artifact.
- **Controls:** the article verdict is computed on the dominant candidate itself.
- **Decision zone:** layers k > L·(1 − 0.30) (for 40 layers: 29–40), with a small tolerance against floating-point error. Its mean is a reported column.
- **QA is chosen by the registry's base/instruct field**, not by the presence of a chat template: Qwen 2.5's base tokenizer ships one.
- **No `trust_remote_code`:** every model in the registry is supported natively by transformers.
- **Tuned Lens sequences start with the beginning-of-sequence token** (the package used an end-of-text token); translators are float32 by default.
- **Mixed models use non-dominant rows only**, so the mirrored values aren't counted twice.
- **Permutation test** is exact up to 16 units, otherwise 100,000 Monte Carlo sign flips.
- **Palette** is Okabe-Ito with a dash pattern per language. The Fig. 5 bars show the range across models (descriptive).
- **Model keys** in the RUNNING guides are now the registry keys (`llama`, `llama_base`, …); the drafts used names like `llama-instruct`.

## Bugs found during the revision

- **The plan's Phase 0 list:** double norm, missing Gemma 2 softcap, substring matching, zero-padded smoothing, unaccented prompts, mixed entity sets, stale counts.
- **YAML read the Portuguese contraction "no" as `false`.** It's quoted now, and validation rejects any non-text word.
- **transformers 5.7 fills unset generation settings from the model's own config** (e.g. Llama's top-p 0.9). Every setting is now explicit, and a test checks the merged result.
- **Bootstrap CIs depended on the row order** of the input table. Paraphrases are now sorted within each formulation.
- **`datasets` 5.x pulls in huggingface_hub 2.x**, which breaks transformers 5.7. The requirements pin `datasets==4.8.5`.
- **Tuned Lens evaluation crashed** when a language's held-out stream ran out (found by the dry run). It now evaluates what is available.
- **Training-memory estimate corrected:** gradients and momentum add about 6 GB for Llama and Gemma. On 24 GB cards, use bf16 translators.

## Where things are

| File | What |
| --- | --- |
| `README.md` | full pipeline, options, models, runtime and memory estimates, code map |
| `RUNNING_1GPU.md`, `RUNNING_multiGPU.md` | step-by-step commands for the real runs (updated; untracked, as agreed) |
| `reports/verified_values.md` | every value confirmed by the author |
| `reports/phase1_review.md`, `prompts_review.md`, `behavioral_review.md` | review documents for the pending confirmations (regenerate with `python scripts/make_review_doc.py`) |
| `dry_run/dry_run_report.json` | last dry run: time per step, workload counts, checks (not versioned) |
