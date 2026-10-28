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

Models, facts, languages, formulations, stake languages, decision zone and smoothing windows are set in `config/experiment_config.py`.
