# linguistic-bias-in-LLM
Linguistic Bias in LLM Outputs

> The pipeline is being revised (see `REVISION_PLAN.md`). This is the interim pipeline after Phase 0; the full pipeline and runtimes are documented at the end of the revision (Phase 8).

## Execution — Step by Step

```bash
# 0. Setup
pip install -r requirements.txt
export HF_TOKEN="hf_your_token_here"
pytest                                 # CPU, tiny random models

# 1. Verify (fast — tokenizers only)
python verify_setup.py

# 2. Run experiments (one model at a time; resumes from checkpoint)
#    Each run starts with the lens sanity check and stops if it fails.
python run_single_model.py mistral
python run_single_model.py llama
python run_single_model.py qwen
python run_single_model.py gemma

# 3. Check integrity (expected run count derived from config)
python check_results.py

# 4. Metrics (reads raw results, never modifies them)
python analyze.py                      # results/metrics_computed.{json,csv}

# 5. Figures
python make_figures.py                 # figures/
```

Configuration lives in YAML files, read by `config/loader.py`:

- `config/experiment.yaml`: languages, prompt types, decision zone, smoothing, thresholds, models;
- `config/facts.yaml`: facts, candidates, single-name sets per language (Russian by case), stakes, narrative indicators;
- `config/prompts.yaml`: every prompt (3 paraphrases per type) with its review and grammar flags;
- `config/baseline_prompts.yaml`: neutral prompts (unrelated topics, same slot shape as each real paraphrase) for the common-token baseline.

The Phase 2 lens lives in `logit_lens/lens.py` (pluggable `Lens`; `LogitLens` now, Tuned Lens in Phase 4) and `logit_lens/measure.py` (`LensRunner`: exact per-candidate probabilities over distinct token IDs, KL to the final distribution, top-20 for inspection, neutral baseline). The run script that writes the results table comes later in Phase 2.

`python scripts/make_review_doc.py` regenerates `reports/phase1_review.md` and `reports/prompts_review.md` from them. Runs refuse facts that are not `status: confirmed`, prompts that still need native review, and an unconfirmed baseline.
