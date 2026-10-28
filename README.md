# linguistic-bias-in-LLM
Linguistic Bias in LLM Outputs

> The pipeline is being revised (see `REVISION_PLAN.md`). This is the interim pipeline after Phase 2; the full pipeline and runtimes are documented at the end of the revision (Phase 8).

## Execution — Step by Step

```bash
# 0. Setup
pip install -r requirements.txt
export HF_TOKEN="hf_your_token_here"
pytest                                 # CPU, tiny random models

# 1. Verify (fast — tokenizers only)
python verify_setup.py

# 2. Measure (one model at a time; resumes from checkpoint)
#    Refuses to run until facts, prompts, baseline and forcing words are confirmed.
#    Starts with the lens sanity check and stops if it fails.
python run_lens.py mistral
python run_lens.py llama
python run_lens.py qwen
python run_lens.py gemma
python monitor.py                      # progress, in another terminal

# 3. Check integrity (expected prompts and units derived from config)
python check_results.py

# 4. Metrics (reads results/lens/*, never modifies them)
python analyze.py                      # results/metrics/

# 5. Figures
python make_figures.py                 # figures/
```

Configuration lives in YAML files, read by `config/loader.py`:

- `config/experiment.yaml`: languages, prompt types, decision zone, smoothing, thresholds, models;
- `config/facts.yaml`: facts, candidates, single-name sets per language (Russian by case), stakes, narrative indicators;
- `config/prompts.yaml`: every prompt (3 paraphrases per type) with its review and grammar flags;
- `config/baseline_prompts.yaml`: neutral prompts (unrelated topics, same slot shape as each real paraphrase) for the common-token baseline;
- `config/forcing.yaml`: articles, plural nouns and contracted prepositions for the article-confound test.

`python scripts/make_review_doc.py` regenerates `reports/phase1_review.md` and `reports/prompts_review.md` from them. Runs refuse facts that are not `status: confirmed`, prompts that still need native review, an unconfirmed baseline and unconfirmed forcing words.

## Measurement (Phase 2)

- `logit_lens/lens.py`: pluggable `Lens` (the model's own head plus a per-layer translation); `LogitLens` now, Tuned Lens in Phase 4. Every forward pass checks that the last layer reproduces the model's logits.
- `logit_lens/measure.py`: `LensRunner`, exact per-candidate probability over distinct token IDs (names on F1–F3/QA, indicators on place/artifact), KL(lens_k ‖ final), top-20 for inspection, neutral baseline. Formats: completion (raw prompt) and QA (chat template, instruct only).
- `logit_lens/confound.py`: article-confound passes (forced article, then plural noun; contracted prepositions for blocked prompts; shared prefixes of split indicators).
- `run_lens.py` writes `results/lens/{model}/`: `probs`, `baseline`, `forced`, `layers`, `token_sets` (Parquet) and `manifest.json`.
- `analyze.py` writes `results/metrics/`: CS_e = P(e) − P(dominant), baseline-adjusted CS, k*, CS*, suppression, decision-zone mean, article verdict, and ΔCS_e(l) = CS_e(l) − CS_e(en) on paraphrase means.
